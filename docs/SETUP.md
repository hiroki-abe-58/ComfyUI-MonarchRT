# Setting up a runtime

The nodes run the upstream pipeline in a separate Python environment (the
*runtime*), so nothing is installed into ComfyUI's own Python. Two layouts
are supported:

| ComfyUI host | Runtime | Config `kind` | Tested with real weights |
| --- | --- | --- | --- |
| Windows 10/11 | a WSL2 Linux distribution on the same PC | `wsl` | yes (Windows 11 + WSL2 Ubuntu 24.04, RTX 5090) |
| Linux | a venv on the same machine | `posix` | CPU tests only |

Native Windows runtimes (no WSL2) are not supported: the upstream kernels need
Triton, flash-attn and FlashInfer builds that this project has not verified on
Windows.

Below, `/opt/monarchrt` is an example location inside the Linux side; any
folder works. Commands run inside WSL2 (or on the Linux host). The reference
versions are listed in `runtime/requirements-reference.txt`.

## 1. GPU and drivers

`nvidia-smi` must work inside the distribution (on WSL2 this comes from the
Windows NVIDIA driver; do not install a Linux display driver inside WSL). The
reference run used driver 595.95 and CUDA 12.8 wheels on an RTX 5090
(compute capability 12.0).

## 2. Python environment

Python 3.10 (the upstream code pins `numpy==1.24.4`). With
[uv](https://docs.astral.sh/uv/) (any Python 3.10 venv works):

```sh
uv venv --python 3.10 /opt/monarchrt/venv
uv pip install --python /opt/monarchrt/venv/bin/python \
  --index-url https://download.pytorch.org/whl/cu128 torch==2.8.0+cu128 torchvision==0.23.0+cu128
# flash-attn 2.8.3 and flashinfer-jit-cache 0.6.3+cu128 wheels: download from the GitHub releases
# listed in requirements-reference.txt, check the sha256, then install the local files
uv pip install --python /opt/monarchrt/venv/bin/python ./flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp310-cp310-linux_x86_64.whl \
  ./flashinfer_jit_cache-0.6.3+cu128-cp39-abi3-manylinux_2_28_x86_64.whl
uv pip install --python /opt/monarchrt/venv/bin/python -r requirements-reference.txt
```

Why these versions: upstream cross-attention calls flash-attn 2 directly (no
SDPA fallback), and the official flash-attn 2.8.3 wheels exist for torch 2.8;
`flashinfer-jit-cache` ships prebuilt kernels, so no CUDA toolkit is needed.

## 3. A C compiler for Triton

Triton compiles a small C launcher the first time each kernel runs. If the
distribution has `gcc` (`sudo apt install build-essential`), nothing else is
needed. Without system packages, the `ziglang` wheel works as the compiler;
the reference runtime used this wrapper as `CC` (it maps GNU ld's
`-l:libcuda.so.1` to the WSL driver library path, which zig's linker does not
accept):

```sh
#!/bin/sh
# /opt/monarchrt/bin/cc-for-triton
set -e
n=$#; i=0
while [ $i -lt $n ]; do
  a="$1"; shift
  case "$a" in
    -l:*) f="${a#-l:}"
          for d in /usr/lib/wsl/lib /opt/monarchrt/venv/lib/python3.10/site-packages/triton/backends/nvidia/lib; do
            [ -e "$d/$f" ] && { a="$d/$f"; break; }
          done ;;
  esac
  set -- "$@" "$a"; i=$((i+1))
done
exec /opt/monarchrt/venv/bin/python -m ziglang cc "$@"
```

(`uv pip install ziglang==0.15.1` into the same venv.)

## 4. Upstream code (pinned)

```sh
git clone https://github.com/Infini-AI-Lab/MonarchRT /opt/monarchrt/MonarchRT
git -C /opt/monarchrt/MonarchRT checkout 34867c041ff7d70c699a9149f84857058d8ada92
```

Do not `pip install` it; the runner imports it from this folder unmodified.
The Doctor node checks the sha256 of the upstream files this integration
relies on.

## 5. Weights (pinned revisions, about 23 GB)

```sh
cd /opt/monarchrt/models
hf download Wan-AI/Wan2.1-T2V-1.3B --revision 37ec512624d61f7aa208f7ea8140a131f93afc9a \
  --local-dir wan_models/Wan2.1-T2V-1.3B \
  config.json diffusion_pytorch_model.safetensors models_t5_umt5-xxl-enc-bf16.pth Wan2.1_VAE.pth \
  google/umt5-xxl/special_tokens_map.json google/umt5-xxl/spiece.model \
  google/umt5-xxl/tokenizer.json google/umt5-xxl/tokenizer_config.json LICENSE.txt README.md
hf download gdhe17/Self-Forcing checkpoints/self_forcing_dmd.pt \
  --revision 2f8b779212da279d212c22a509b66ad6552f350e --local-dir .
```

Only the files above are needed. Run the Doctor node with `verify_sha256`
once to compare every file with the hashes in `runtime/monarchrt_doctor.py`.
Keeping the models on the Linux file system (not under `/mnt/c` etc.) makes
loading faster; both work.

## 6. Register the runtime in ComfyUI

Copy `examples/monarchrt.runtimes.example.json` to
`<ComfyUI>/user/monarchrt.runtimes.json` (or point the environment variable
`COMFYUI_MONARCHRT_CONFIG` to it before starting ComfyUI), keep one entry and
fill in your paths:

- `distro`: the name shown by `wsl -l -v` (kind `wsl` only);
- `python`, `upstream_dir`, `models_dir`, `checkpoint`: Linux paths;
- `env`: only `CC`, `TRITON_CACHE_DIR`, `TORCHINDUCTOR_CACHE_DIR`,
  `XDG_CACHE_HOME`, `HF_HOME`, `FLASHINFER_WORKSPACE_BASE`,
  `CUDA_VISIBLE_DEVICES` are accepted. Point the cache folders at a
  persistent location: Triton keeps compiled kernels there;
- `offload_text_encoder`: `true` on GPUs with less than 40 GB (what upstream
  `inference.py` does automatically). The 11 GB UMT5-XXL encoder then stays
  in CPU memory and is streamed to the GPU for the prompt encoding only;
- `timeout_minutes`, optional `jobs_dir` (a folder on a local drive; default
  `<ComfyUI temp>/monarchrt`), optional `wsl_mount_root` (default `/mnt/`).
- `backend` (optional): `persistent` keeps one warm worker process between
  queue jobs, `one-shot` (the default when the key is missing) starts a
  process per job. The Runtime node can override it per workflow.
- `worker_idle_seconds` (optional, 10-86400, default 300): a persistent worker
  exits after this long without a job and frees its GPU and RAM.

Restart ComfyUI, add **MonarchRT Doctor**, pick the runtime and queue it.

## Memory and first-run behaviour

- RAM: the reference runtime peaked at about 21 GB resident in WSL2 (31 GB
  WSL memory limit). The runner builds the T5 encoder directly in bf16 from a
  memory-mapped file instead of upstream's fp32 copy (bit-identical result).
- First run after setup: Triton compiles and autotunes the Monarch kernels for
  each KV-cache length, about 1.5-2 minutes per length and 7 lengths per
  video (h_reduce 1 and 2 are tuned separately). Compiled kernels are cached
  in `TRITON_CACHE_DIR`; later processes still re-run the timing part of the
  autotune once per length (about 20 s each; upstream README explains why).
  Generating several videos in one job (`videos` input) pays this once.
