"""Official HF Ouro per-cycle readout; no SGLang, KV cache, early gate or speed claim."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from .readout_metrics import measure_readouts, summarize_readouts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--input', type=Path, default=Path(__file__).with_name('readout_prompts.jsonl'))
    parser.add_argument('--output', type=Path, required=True, help='New directory')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--limit', type=int, default=20)
    parser.add_argument('--max-new-tokens', type=int, default=32)
    args = parser.parse_args()
    if args.limit < 0 or args.max_new_tokens < 1:
        parser.error('Require limit >= 0 and max-new-tokens > 0')
    root = args.model.expanduser().resolve()
    for name in ['config.json', 'configuration_ouro.py', 'modeling_ouro.py']:
        if not (root/name).is_file():
            parser.error(f'Missing {root/name}; download the complete official snapshot including Python source')
    raw = json.loads((root/'config.json').read_text())
    if raw.get('model_type') != 'ouro' or raw.get('total_ut_steps') != 4:
        parser.error('This diagnostic requires the four-cycle Ouro checkpoint')
    prompts = [json.loads(s) for s in args.input.read_text().splitlines() if s.strip()]
    if args.limit:
        prompts = prompts[:args.limit]
    if not prompts or any(not isinstance(p.get('prompt'), str) or not p['prompt'].strip() for p in prompts):
        parser.error('Input requires nonempty prompt strings')
    ids = [str(p.get('id', i)) for i,p in enumerate(prompts)]
    if len(set(ids)) != len(ids):
        parser.error('Duplicate sample IDs')
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer
    args.output.mkdir(parents=True, exist_ok=False)
    settings = {k:str(v) if isinstance(v, Path) else v for k,v in vars(args).items()}
    settings.update(torch_version=torch.__version__, transformers_version=transformers.__version__,
                    use_cache=False, full_depth_cycles=4, exit_at_step=3,
                    prompt_style='raw completion without chat template',
                    selected_samples=len(prompts), purpose='readout diagnostic, not LoopSpec pipeline reproduction or speed benchmark',
                    source_sha256={name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in ['config.json','modeling_ouro.py','configuration_ouro.py']})
    settings['revision'] = subprocess.check_output(['git','rev-parse','HEAD'], cwd=Path(__file__).resolve().parents[1], text=True).strip()
    def write_settings():
        (args.output/'settings.json').write_text(json.dumps(settings, indent=2), encoding='utf-8')
    write_settings()
    tokenizer = AutoTokenizer.from_pretrained(str(root), trust_remote_code=True, local_files_only=True)
    model, loading = AutoModelForCausalLM.from_pretrained(
        str(root), trust_remote_code=True, local_files_only=True,
        torch_dtype=torch.bfloat16, attn_implementation='sdpa', output_loading_info=True)
    settings['loading_info'] = loading
    write_settings()
    if any(loading.get(k) for k in ['missing_keys','unexpected_keys','mismatched_keys','error_msgs']):
        raise RuntimeError('Checkpoint loading mismatch; inspect settings.json')
    model = model.to(args.device).eval().requires_grad_(False)
    if not hasattr(model.model, 'norm') or model.model.total_ut_steps != 4:
        raise RuntimeError('Unexpected official Ouro implementation; expected model.norm called once per cycle')
    settings['device_name'] = torch.cuda.get_device_name(torch.device(args.device)) if args.device.startswith('cuda') else 'cpu'
    settings['layers_per_cycle'] = model.config.num_hidden_layers
    settings['effective_layer_applications'] = 4*model.config.num_hidden_layers
    write_settings()
    eos = model.config.eos_token_id
    eos_ids = set(eos if isinstance(eos, list) else [eos]) if eos is not None else set()
    all_rows = []
    with (args.output/'tokens.csv').open('w', newline='', encoding='utf-8') as tf, (args.output/'generations.jsonl').open('w', encoding='utf-8') as gf:
        writer = None
        for number, (sample, sample_id) in enumerate(zip(prompts, ids), 1):
            print(f'[{number}/{len(prompts)}] {sample_id}', flush=True)
            input_ids = tokenizer(sample['prompt'], return_tensors='pt')['input_ids'].to(args.device)
            generated = []
            for position in range(args.max_new_tokens):
                captured = []
                def hook(_module, _args, output):
                    captured.append(output[0, -1].detach().clone())
                handle = model.model.norm.register_forward_hook(hook)
                try:
                    with torch.inference_mode():
                        out = model(input_ids=input_ids, attention_mask=torch.ones_like(input_ids),
                                    use_cache=False, logits_to_keep=1,
                                    exit_at_step=3, use_weighted_exit=False)
                        if len(captured) != 4:
                            raise RuntimeError(f'Expected four normalized cycle outputs; got {len(captured)}')
                        logits = [model.lm_head(hidden).float().unsqueeze(0) for hidden in captured]
                        final = out.logits[:, -1].float()
                        token = int(final.argmax(-1).item())
                        if int(logits[-1].argmax(-1).item()) != token:
                            raise RuntimeError('Final-cycle hooked readout differs from explicit full-depth output')
                        for cycle in range(4):
                            row = measure_readouts(logits[cycle], final, [token])[0]
                            a, b = captured[cycle].float(), captured[-1].float()
                            row.update(id=sample_id, category=sample.get('category','text'),
                                       position=position, stage=f'cycle{cycle+1}', is_eos=token in eos_ids,
                                       early_hidden_rms=float(a.square().mean().sqrt().item()),
                                       final_hidden_rms=float(b.square().mean().sqrt().item()),
                                       hidden_cosine=float(torch.nn.functional.cosine_similarity(a,b,dim=0).item()))
                            for field in ['reference_id','early_id','final_id']:
                                row[field.replace('_id','_text')] = tokenizer.decode([row[field]], skip_special_tokens=False)
                            row['early_top10_text'] = [tokenizer.decode([i], skip_special_tokens=False) for i in row['early_top10_ids']]
                            if writer is None:
                                writer = csv.DictWriter(tf, fieldnames=list(row)); writer.writeheader()
                            writer.writerow({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(dict,list)) else v for k,v in row.items()})
                            all_rows.append(row)
                finally:
                    handle.remove()
                generated.append(token)
                if token in eos_ids:
                    break
                input_ids = torch.cat([input_ids, input_ids.new_tensor([[token]])], dim=1)
            tf.flush()
            gf.write(json.dumps({**sample,'id':sample_id,'token_ids':generated,
                                 'text':tokenizer.decode(generated,skip_special_tokens=True)},ensure_ascii=False)+'\n'); gf.flush()
            summary = summarize_readouts(all_rows)
            (args.output/'readout_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    (args.output/'COMPLETE.json').write_text(json.dumps({'samples':len(prompts),'generated_tokens':len(all_rows)//4}),encoding='utf-8')
    print(json.dumps(summarize_readouts(all_rows),indent=2))


if __name__ == '__main__':
    main()
