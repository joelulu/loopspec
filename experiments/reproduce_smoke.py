"""Single-GPU sequential AR / LoopSpec smoke, using the original server.

Run with the repository's pinned Python environment. This is a tiny paired
correctness/speed check, not a reproduction of the paper's full benchmark.
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


MODES = {"baseline": [], "single": ["--loopspec", "--first", "1"],
         "dual": ["--loopspec", "--first", "1", "--second", "2"]}


def consume_stream(lines):
    """Accept either cumulative or incremental output_ids; never retokenize text."""
    ids, text, meta = [], "", None
    started = time.perf_counter()
    first = None
    done = False
    for raw in lines:
        line = raw.decode("utf-8").strip()
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data == "[DONE]":
            done = True
            break
        event = json.loads(data)
        if "error" in event:
            raise RuntimeError(str(event["error"]))
        if isinstance(event.get("meta_info"), dict):
            meta = event["meta_info"]
        if first is None and meta and meta.get("completion_tokens", 0) > 0:
            first = time.perf_counter() - started
        event_text = event.get("text")
        cumulative = True
        if "output_ids" in event:
            new = event["output_ids"]
            if not isinstance(new, list) or any(type(t) is not int for t in new):
                raise RuntimeError("Invalid output_ids from server")
            count = meta.get("completion_tokens") if meta else None
            if len(new) == count:
                ids = new
            elif len(ids) + len(new) == count:
                ids.extend(new)
                cumulative = False
            else:
                raise RuntimeError("output_ids not aligned with completion_tokens")
        if isinstance(event_text, str):
            text = event_text if cumulative else text + event_text
    if not done or meta is None or type(meta.get("completion_tokens")) is not int:
        raise RuntimeError("Incomplete generation stream")
    count = meta["completion_tokens"]
    return {"text": text, "token_ids": ids if len(ids) == count else None,
            "completion_tokens": count, "meta_info": meta,
            "client_stream_ttft_seconds": first}


def generate(base_url, prompt, budget, rid, timeout):
    payload = {"text": prompt, "rid": rid, "stream": True,
               "sampling_params": {"temperature": 0, "top_p": 1.0,
                                   "max_new_tokens": budget, "stream_interval": 1}}
    request = urllib.request.Request(base_url+"/generate", data=json.dumps(payload).encode(),
                                     headers={"Content-Type":"application/json"})
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=timeout) as response:
        result = consume_stream(response)
    result["client_request_seconds"] = time.perf_counter()-started
    # Stream-only timing excludes HTTP setup, so do not label it end-to-end TTFT.
    result.pop("client_stream_ttft_seconds", None)
    return result


def stop_server(process):
    """Terminate the entire process group, including scheduler workers."""
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=20)
    except subprocess.TimeoutExpired:
        pass
    # The parent can exit while worker descendants remain in its process group.
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait(timeout=10)


def read_metrics(log_path):
    rows = []
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if "generation_metrics " in line:
            rows.append(json.JSONDecoder().raw_decode(line.split("generation_metrics ", 1)[1])[0])
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path, help="Local Ouro-1.4B directory")
    parser.add_argument("--output", type=Path, required=True, help="New directory")
    parser.add_argument("--input", type=Path, default=Path(__file__).with_name("smoke_prompts.jsonl"))
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--linear-backend", choices=("auto", "torch"),
                        default=os.environ.get("LOOPSPEC_LINEAR_BACKEND", "auto"),
                        help="torch skips the optional TinyGEMM patch in every mode")
    parser.add_argument("--port", type=int, default=30000)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--memory-fraction", type=float, default=.55)
    parser.add_argument("--timeout", type=int, default=1800)
    args = parser.parse_args()
    if args.linear_backend not in ("auto", "torch"):
        parser.error("LOOPSPEC_LINEAR_BACKEND must be auto or torch")
    if args.max_new_tokens < 2 or args.timeout < 1 or not 0 < args.memory_fraction < 1:
        parser.error("Require max-new-tokens >= 2, timeout > 0, memory-fraction in (0,1)")
    if not 1 <= args.port <= 65535:
        parser.error("Invalid port")
    model = args.model.expanduser().resolve()
    cfg = json.loads((model/"config.json").read_text())
    if cfg.get("model_type") != "ouro" or cfg.get("total_ut_steps") != 4:
        parser.error("This preset requires an Ouro checkpoint with total_ut_steps=4")
    prompts = [json.loads(line) for line in args.input.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not prompts or any(not isinstance(p.get("prompt"), str) or not p["prompt"].strip() for p in prompts):
        parser.error("Input needs nonempty prompt fields")
    ids = [str(p.get("id", i)) for i,p in enumerate(prompts)]
    if len(set(ids)) != len(ids):
        parser.error("Input ids must be unique")
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", args.port))  # Fail before starting if occupied.
    args.output.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[1]
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": args.gpu,
           "LOOPSPEC_LINEAR_BACKEND": args.linear_backend}
    base_url = f"http://127.0.0.1:{args.port}"
    settings = {k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}
    settings.update(model=str(model), python=sys.version, modes=list(MODES),
                    samples=len(prompts), warmup_requests_per_mode=1, prompt_style="raw completion; no chat template",
                    revision=subprocess.check_output(["git","rev-parse","HEAD"],cwd=root,text=True).strip())
    (args.output/"settings.json").write_text(json.dumps(settings,indent=2),encoding="utf-8")
    results = {}
    for mode, flags in MODES.items():
        command = [sys.executable,"-m","sglang_recurrent.server","--model",str(model),
                   "--host","127.0.0.1","--port",str(args.port),"--max-running-requests","1",
                   "--mem-fraction-static",str(args.memory_fraction),"--random-seed","1234",
                   "--stream-interval","1",*flags]
        print(f"Starting {mode}",flush=True)
        log_path = args.output/f"{mode}.server.log"
        collected = []
        with log_path.open("w",encoding="utf-8") as log:
            process = subprocess.Popen(command,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            try:
                deadline=time.monotonic()+args.timeout
                while True:
                    if process.poll() is not None:
                        raise RuntimeError(f"{mode} server exited; see {log_path}")
                    try:
                        with urllib.request.urlopen(base_url+"/health",timeout=2) as response:
                            ready=response.status==200
                    except (OSError,urllib.error.URLError):
                        ready=False
                    if ready:
                        break
                    if time.monotonic()>deadline:
                        raise TimeoutError(f"{mode} startup timeout; see {log_path}")
                    time.sleep(1)
                generate(base_url,prompts[0]["prompt"],min(16,args.max_new_tokens),f"smoke-{mode}-warmup",args.timeout)
                with (args.output/f"{mode}.jsonl").open("w",encoding="utf-8") as stream:
                    for i,prompt in enumerate(prompts):
                        print(f"{mode} [{i+1}/{len(prompts)}] {ids[i]}",flush=True)
                        result={"id":ids[i],"prompt":prompt["prompt"],**generate(base_url,prompt["prompt"],args.max_new_tokens,f"smoke-{mode}-{i}",args.timeout)}
                        collected.append(result)
                        stream.write(json.dumps(result,ensure_ascii=False)+"\n");stream.flush()
            finally:
                stop_server(process)
        metrics=read_metrics(log_path)
        if len(metrics)!=len(prompts)+1:
            raise RuntimeError(f"Expected {len(prompts)+1} generation_metrics including warmup, got {len(metrics)}; see {log_path}")
        measured=metrics[1:]
        for response,metric in zip(collected,measured):
            if metric["decode_tokens"]!=max(0,response["completion_tokens"]-1):
                raise RuntimeError("Server timing rows do not match request token counts")
        seconds=sum(m["decode_seconds"] for m in measured)
        count=sum(m["decode_tokens"] for m in measured)
        results[mode]={"responses":collected,"server_metrics":measured,
                       "decode_tokens":count,"decode_seconds":seconds,
                       "decode_tok_s":count/seconds if count and seconds>0 else None}
    comparison=[]
    for mode in ("single","dual"):
        pairs=[]
        for baseline,other in zip(results["baseline"]["responses"],results[mode]["responses"]):
            match=(baseline["token_ids"]==other["token_ids"]) if baseline["token_ids"] is not None and other["token_ids"] is not None else None
            pairs.append({"id":baseline["id"],"token_ids_match":match,"text_match":baseline["text"]==other["text"]})
        a,b=results["baseline"]["decode_tok_s"],results[mode]["decode_tok_s"]
        comparison.append({"mode":mode,"pairs":pairs,"speedup":b/a if a and b and all(p["token_ids_match"] is True for p in pairs) else None})
    summary={"scope":"tiny greedy smoke; server decode timing excludes first token and prefill",
             "results":results,"comparison":comparison}
    (args.output/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")
    if not all(p["token_ids_match"] is True for c in comparison for p in c["pairs"]):
        raise RuntimeError("Token equality failed or output_ids unavailable; see summary.json. Text equality alone is insufficient.")
    (args.output/"COMPLETE.json").write_text(json.dumps({"samples":len(prompts),"modes":list(MODES)}),encoding="utf-8")
    print(json.dumps(comparison,ensure_ascii=False,indent=2))


if __name__ == "__main__":
    main()
