# Ouro 单卡最小复现

本入口顺序启动原始 server：AR、LoopSpec 单 proposal、LoopSpec 双 proposal。
同一张 GPU 每次只保留一个 server；原有模型、pipeline 和采样实现不作修改。
仅用于环境/greedy一致性/计时流程检查，不是完整论文 benchmark，也不评测任务准确率。

## 环境与权重

使用独立 Linux x86-64 / Python 3.11 环境；不要复用 LoopVL 的环境。
优先沿用上游锁定依赖，而不是将服务器已有的 SGLang 强行覆盖：

```bash
uv sync --locked --no-dev
uv run --locked --no-dev hf download ByteDance/Ouro-1.4B --local-dir models/Ouro-1.4B
```

上游固定 torch 2.11.0、SGLang 0.5.14、Transformers 5.8.1、
flashinfer cu13；需要兼容 CUDA 的 NVIDIA 驱动。安装/启动失败时保留完整日志，
不要为了绕过错误随意升级单个依赖。上游 Dockerfile 是另一种环境路径。

## 一条命令完成三个配置

```bash
uv run --locked --no-dev python experiments/reproduce_smoke.py \
  --model models/Ouro-1.4B --gpu 0 --max-new-tokens 128 \
  --output results/smoke_ouro
```

输出目录必须不存在，防止覆盖之前的实验。
脚本依次测试 `baseline`、`--first 1`、`--first 1 --second 2`，每个配置先
执行1次16-token以内的 warmup，再运行 `smoke_prompts.jsonl` 的3条请求。
总计3条不同测试 prompt、9次正式请求、3次warmup；不下载数学/代码测试集。
这是 raw completion 提示，不自动套 chat template。

可通过 `--input your_prompts.jsonl` 换成自己的 `{id,prompt}` JSONL。
`--gpu` 指定当前进程可见的设备编号/UUID，`--memory-fraction` 默认0.55。
每个配置固定单请求、greedy、同一输出预算和 server seed 1234。
不要同时在选定 GPU 上运行另一个本实验 server。

## 查看结果

- `*.server.log`：原始启动日志、pipeline计数和 `generation_metrics`。
- `baseline.jsonl` / `single.jsonl` / `dual.jsonl`：逐请求文本、原始 token IDs、客户端耗时和返回 metadata。
- `summary.json`：逐样本 token 精确匹配，以及 server decode tok/s 和相对 AR speedup。
- `settings.json`：实际参数、git revision 和 Python 版本。
- `COMPLETE.json`：三个配置全部通过，且每条token序列都与AR完全一致。

速度从上游 server 的 `generation_metrics` 汇总：排除首个completion token、
prefill和warmup；不将客户端HTTP耗时当成decode时间。要求日志行数和token数与
请求一一匹配，否则停止。token不一致或HTTP接口没有返回 `output_ids` 时，
记录诊断并失败，**不会用重新分词文本代替原始token，也不会输出有效speedup**。
只有极短输出时没有有效decode区间，吞吐和speedup记null。

Ctrl-C/请求失败会终止脚本启动的server进程组；日志和已完成请求保留。
完整数学benchmark继续使用上游 `benchmark.py` 和原YAML，不用本smoke代替。

## 无GPU协议测试

```bash
python -m unittest experiments.test_reproduce_smoke -v
```

## 隔离 TinyGEMM / CUTLASS DSL 启动错误

可在上述命令添加 `--linear-backend torch`，跳过本仓库的 FlashInfer
TinyGEMM 安装探测和线性层替换，三种模式统一保留 SGLang 原有线性层实现。
这不保证关闭 SGLang/FlashInfer 其他位置使用的 CUTLASS 内核；仍然报错时需查看完整调用栈。
默认 `auto` 保持上游行为。参数记录在 `settings.json`，比较速度时使用相同设置；
关闭 TinyGEMM 的性能不能直接当作上游优化配置的复现结果。
直接启动 server 时，可在启动前设置 `export LOOPSPEC_LINEAR_BACKEND=torch`。
