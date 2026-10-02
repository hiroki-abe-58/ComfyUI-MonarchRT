# Testing

## CPU tests (CI and local)

```sh
python -m pip install pytest==9.1.1 ruff==0.16.10
COMFYUI_PATH=/path/to/ComfyUI python -m pytest -q -rs
```

No GPU and no weights: `tests/fake_runtime/monarchrt_job.py` stands in for the
upstream pipeline. It validates jobs with the real runner's `load_job`, writes
real 81-frame 832x480 MP4 files and can simulate failures and hangs (with a
grandchild process). What is covered:

- runtime config validation (unknown keys, relative paths, `..`, distro names,
  environment allowlist, limits) and the config file location;
- argv construction (fixed list, `wsl.exe --exec`, no shell), launcher
  environment without secrets (`WSLENV` emptied), WSL path translation;
- job creation and the runner's own job validation (tampered jobs rejected),
  the runner's attention-dispatch check (no silent fallback);
- result validation (path confinement, sha256, geometry);
- real subprocesses: success with progress, failure reporting, cancel and
  timeout. On Linux these also prove that the runner's process group,
  including a grandchild, is gone afterwards, that the kill helper refuses a
  pid that does not belong to the job, and that it stops orphans left after
  the runner was killed from outside;
- inside a real ComfyUI checkout (CPU): node registration, no heavy imports
  (`triton`, `flash_attn`, `flashinfer`, upstream modules) at import time,
  `validate_prompt` for good and bad workflows, the Generate node returning
  ComfyUI `VIDEO` objects (decoded back: 81 frames, 16 fps, 832x480), and
  ComfyUI interrupts mapped to a job stop.

Tests that need a ComfyUI checkout are marked `comfy` and fail (not skip) when
`COMFYUI_PATH` is missing. Three process tests that read `/proc` are
POSIX-only; on Windows hosts that code runs inside WSL, and the tests run on
the Ubuntu CI leg (and were run in a WSL2 venv locally).

## Persistent worker tests (CPU)

`tests/test_worker.py` runs the real `runtime/monarchrt_worker.py` (frozen into
a session folder exactly like in production) with a fake engine
(`tests/fake_runtime/fake_monarchrt_worker.py`) and checks: three queue jobs
on one worker (same pid, one model load), profile switches without restart,
restart on identity change (old worker gone), status and unload, the idle
timeout, cooperative cancel (worker stays), cancel that is not confirmed
(worker stopped), timeout, a job error (worker stays) vs a CUDA-type error
(worker replaced), a worker crash (the job fails once and is not retried),
concurrent jobs serialised, busy / duplicate / foreign-path requests refused,
an exception from the progress callback (job cancelled), and - on Linux -
that the worker stops when its parent process dies while idle and while busy.

## GPU end-to-end (maintainer, real runtime)

`scripts/gpu_e2e_persistent.py` does the same through ComfyUI's HTTP API with
the real runtime (8+ separate queue jobs, output hashes against one-shot
references, unload, cancel, worker and ComfyUI crashes, errors, idle timeout,
memory after every job); see `docs/BENCHMARKS.md#persistent-worker-v020`.

`scripts/gpu_e2e.py` starts its own ComfyUI on a free `127.0.0.1` port and
uses the HTTP API (`/prompt`, `/history`, `/interrupt`, `/object_info`) with a
real runtime:

| Step | Pass criteria |
| --- | --- |
| generate | MonarchRT job with 2 videos -> `SaveVideo`; both MP4 files fully decoded (81 frames, 832x480, 16 fps); report shows 1050 Monarch Triton calls and 0 dense self-attention calls per video |
| doctor | Doctor (with kernel check) reports `ok` |
| cancel | `/interrupt` during generation -> `execution_interrupted`, no process of the job left in WSL, GPU memory back to idle |
| timeout | runtime with `timeout_minutes: 1` -> `execution_error`, nothing left, GPU memory back to idle |
| error | runtime with a missing checkpoint -> `execution_error` with the runner's message, nothing left |

The results of the published run are in `docs/results/gpu_e2e_report.json`.
During development the ComfyUI process was also killed in the middle of a
job: the runner saw its stdin close, stopped its process group and exited
within about 3 seconds, and GPU memory returned to the idle level.

## GUI check

`scripts/export_gui_workflows.py` (needs Playwright) loads each API workflow
into the real ComfyUI frontend of a running test server, checks that every
node type is known to the frontend and that `graphToPrompt()` reproduces the
API workflow, takes a screenshot, and exports the UI-format files in
`workflows/` (with the seeds of the comparison workflow set to "fixed").

## Clean install

The release was also checked by extracting `git archive` of the tagged commit
into a differently named folder under a fresh ComfyUI `custom_nodes`, running
the CPU tests against it and the GPU end-to-end test through it.
