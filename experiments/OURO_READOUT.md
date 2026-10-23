# Ouro cycle readout diagnostic

This is a small diagnostic of the official Ouro checkpoint's intermediate readouts,
not a reproduction of the LoopSpec serving pipeline, acceptance, or speed.
It imports no SGLang/FlashInfer/CUTLASS code. It uses offline model files including
`configuration_ouro.py` and `modeling_ouro.py` with `trust_remote_code=True`.

Use a separate Python 3.11 environment with torch 2.11.0 and transformers 4.55.0,
the API version declared by the official checkpoint. Do not change the pinned
SGLang environment. Keep your working CUDA compat exports when using cu130.

```bash
uv venv --python 3.11 .venv-ouro-probe
uv pip install --python .venv-ouro-probe/bin/python \
  --index-url https://pypi.tuna.tsinghua.edu.cn/simple \
  torch==2.11.0 transformers==4.55.0 safetensors sentencepiece

.venv-ouro-probe/bin/python -m experiments.ouro_readout_probe \
  --model "$HOME/LMM/lmm_model/Ouro-1.4B" \
  --device cuda:0 --limit 20 --max-new-tokens 32 \
  --output results/ouro_readout_new
```

20 included raw-completion prompts, no chat template or task accuracy evaluation.
At every generated position, the official model runs all four cycles without KV
cache. A hook on its existing norm captures the normalized output of each cycle.
`exit_at_step=3` explicitly selects the final (zero-based fourth) cycle for the
reference token; the learned exit gate does not choose the target. Each early
readout is compared at the same prefix, with the original shared LM head.
The fourth readout must match the official final output. No model weights,
normalization, or recurrence schedule are changed.

Outputs: settings.json (including versions, loading audit and local source hashes),
generations.jsonl, tokens.csv, readout_summary.json, COMPLETE.json.
The summary separates all/content/EOS/first positions for each cycle. Top-1,
Top-5/10 hits, final-token rank and probabilities measure readout agreement, not
speculative acceptance. Hidden RMS and cosine are recorded only as diagnostics.
Cross-model results cannot isolate modality because architecture/training differ.

CPU checks: `python -m unittest experiments.test_readout_metrics -v`.
GPU execution with the actual checkpoint remains required.

Readout schema v2 fixes a diagnostic bug: greedy IDs now use logits.argmax,
matching actual decoding. torch.topk may choose a different ID among tied maxima;
its ordering must not define Top-1 agreement. This could falsely lower Ouro
cycle4 self-agreement and falsely fail LoopVL H1 prefix checks. Top-5/10 still
use actual topk candidate sets; tied boundary membership can be arbitrary.
Rerun diagnostics into a new output directory; weights and inputs are unchanged.
