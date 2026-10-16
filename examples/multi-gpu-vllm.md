# Multi-GPU NVIDIA machine with vLLM

> Status: planning and command generation are covered by tests using
> `examples/hardware/4x-a100-80gb.json` and `8x-h100-80gb.json`. Launching vLLM on real
> multi-GPU hardware has **not** been verified by the authors.

You can plan from any machine with a recorded hardware profile:

```bash
make recommend TASK=coding CONTEXT=32768 CONCURRENCY=16 ENGINE=vllm \
     HARDWARE=examples/hardware/4x-a100-80gb.json EXPLAIN=1
```

On the server itself:

```bash
make doctor
make engine-install ENGINE=vllm
make recommend TASK=agents CONTEXT=32768 CONCURRENCY=16 ENGINE=vllm
make model-pull MODEL=qwen3-32b:bf16           # from the artifact mapped in corporate.yaml
make serve MODEL=qwen3-32b:bf16 ENGINE=vllm CONTEXT=32768 CONCURRENCY=16 PLAN=1   # inspect
make serve MODEL=qwen3-32b:bf16 ENGINE=vllm CONTEXT=32768 CONCURRENCY=16
make serve-test
```

What the planner checks for tensor parallelism

- TP degree is a power of two, ≤ GPU count, and divides the attention-head count.
- Each rank holds `weights / TP` (+2 % replicated tensors) and its KV-head shard; KV
  heads are replicated when TP exceeds the KV-head count.
- The budget per rank is `min(total × gpu_memory_utilization, free − margin)` of the
  **smallest** participating GPU — VRAM is never pooled across devices.
- The smallest TP that serves the requested concurrency in full is preferred; if none
  does, the plan with the most capacity is returned with a warning.

Useful overrides

```bash
make serve MODEL=qwen3-32b:awq ENGINE=vllm TP=4 GPU_UTIL=0.85 PORT=8001 INSTANCE=awq
make serve-status
make serve-stop INSTANCE=awq
```

Running two instances on disjoint GPUs: export `CUDA_VISIBLE_DEVICES` before each
`make serve` and give them different `INSTANCE` and `PORT` values. setup-work does not
yet partition GPUs between instances automatically.
