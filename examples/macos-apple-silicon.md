# macOS on Apple Silicon with llama.cpp

Apple Silicon has unified memory: the GPU can only wire part of RAM (the *Metal
working set*, roughly 2/3 of RAM up to 36 GB and 3/4 above). setup-work uses that
limit as the GPU budget, or the value of `sysctl iogpu.wired_limit_mb` if you raised it.
vLLM is not offered on macOS.

> Status: `doctor`, `hardware` and `recommend` were run on an M4 / 24 GB MacBook Pro.
> Installing and serving from corporate artifacts on macOS was not part of that check.
> A macOS llama.cpp build must be published in `corporate.yaml` (`engines: llama.cpp`).

```bash
make bootstrap DRY_RUN=1            # preview; Homebrew installs (if any) are confirmed
make bootstrap
make doctor

make recommend TASK=coding CONTEXT=16384
#  1  gpt-oss-20b:mxfp4-gguf     llama.cpp  mxfp4   12.3 GiB  full-gpu ...
#  2  qwen2.5-coder-14b:q6_k     llama.cpp  q6_k    14.9 GiB  full-gpu ...

make engine-install ENGINE=llama.cpp          # macOS build from engines['llama.cpp'] in corporate.yaml
make model-pull MODEL=qwen2.5-coder-7b:q4_k_m # ~4.7 GB, from the artifact mapped in corporate.yaml
make serve MODEL=qwen2.5-coder-7b:q4_k_m CONTEXT=16384
make serve-test
make serve-stop
```

Tips

- Close memory-hungry apps: unified memory is shared with everything else.
- `make serve ... FOREGROUND=1` streams the llama.cpp log; Ctrl-C stops it cleanly.
- Use a q4/q5 variant for models close to the budget; `make recommend EXPLAIN=1`
  shows why larger variants were rejected.
