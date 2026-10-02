# Security model

ComfyUI-MonarchRT starts programs outside the ComfyUI process. This page
describes what a workflow can and cannot influence, and how processes are
started and stopped. It is **process isolation of configuration and
arguments, not a sandbox**: the runtime runs with the permissions of the user
account that runs ComfyUI (inside WSL2 as the distribution's default user).

## Who decides what runs

| Decided by | What |
| --- | --- |
| Administrator (local file) | WSL distribution, runtime Python, upstream checkout, model folder, checkpoint, allowlisted runtime environment variables, offload, timeout, job folder |
| This package | The only scripts ever executed: `runtime/monarchrt_job.py`, `runtime/monarchrt_doctor.py`, `runtime/monarchrt_kill.py` |
| Workflow | Data only: runtime id (must exist in the admin file), prompt text, seed, attention profile (`monarch_h2`, `monarch_h1`, `dense`), number of videos (1-4) |

The runtime registration is read from `COMFYUI_MONARCHRT_CONFIG` or
`<ComfyUI user directory>/monarchrt.runtimes.json`. That file sits at the root
of the user directory, outside every per-user folder that ComfyUI's
`/userdata` API reads and writes, so it cannot be created or changed through
the ComfyUI web API. Unknown keys, relative paths, `..` segments, unknown
environment variables and malformed distribution names are rejected.

## How a job is started

- `subprocess.Popen` with an argv list; never `shell=True`, `os.system`,
  `eval` or `exec`. On Windows the launcher is `%SystemRoot%\System32\wsl.exe
  -d <distro> --cd / --exec <python> <script> ...` (no shell inside WSL
  either).
- The workflow's data travels as a schema-checked UTF-8 JSON file in a fresh
  job directory (`job-<time>-<random>`), which must resolve inside the job
  root. The runner validates it again (unknown keys, types, limits).
- The launcher gets a minimal environment: on Windows only the variables
  `wsl.exe` needs plus `WSLENV=""`, so no Windows variable is forwarded into
  Linux; on Linux `PATH`, `HOME`, locale and `LD_LIBRARY_PATH`. The runner
  additionally deletes any variable whose name looks like a credential
  (`TOKEN`, `SECRET`, `PASSWORD`, `API_KEY`, `ACCESS_KEY`, `CREDENTIAL`,
  `WANDB`) and applies only allowlisted settings (`CC`, Triton/Inductor/
  FlashInfer cache folders, `XDG_CACHE_HOME`, `HF_HOME`,
  `CUDA_VISIBLE_DEVICES`). Hugging Face access is forced offline.
- stdout/stderr go to files in the job directory (no pipe that can fill up and
  deadlock); progress comes from `events.jsonl`.
- No package is installed at run time.

## Loading weights

Every `torch.load` in the runtime is forced to `weights_only=True` (upstream
loads the T5 encoder with `weights_only=False`); there is no fallback to
unsafe unpickling. The Self-Forcing EMA weights are loaded with
`strict=True`, after a shape check, and the runner proves that a probe tensor
changed and equals the EMA value.

## Stopping a job

Cancel (ComfyUI interrupt), timeout and errors all end in the same sequence:

1. write `CANCEL` into the job directory and close the runner's stdin; the
   runner kills the other members of its own process group and exits;
2. run `runtime/monarchrt_kill.py` inside the runtime. It reads
   `runner.pid` (pid, process group, start time), checks via `/proc` that the
   pid still belongs to *this* job (command line and start time), and stops
   that process group and the runner's descendants (SIGTERM, then SIGKILL).
   If the runner is already gone, it stops only processes that are still in
   the runner's own group and session and started after it;
3. kill the local launcher process if it is still alive.

Nothing outside the job's own process tree is signalled. The GPU end-to-end
test checks that no process of the job is left and that GPU memory returns to
the idle level after cancel, timeout and error (see docs/TESTING.md).

## Persistent worker (backend `persistent`)

The persistent backend keeps one runtime process alive between queue jobs.
The same rules apply as for one-shot jobs, plus:

- It is started only from a registered runtime, with the same argv and
  environment rules. It runs a copy of `runtime/monarchrt_worker.py` and
  `runtime/monarchrt_job.py` taken into the worker's session folder at
  start-up, so updating the node never changes the code of a running worker.
  The worker's identity (runtime settings, code digest, protocol version)
  is compared before every job; any difference restarts it.
- Communication uses the inherited stdin/stdout pipes only (JSON lines,
  at most 64 KiB per request, no network port). The worker writes its
  replies to a private copy of stdout; everything the upstream code prints
  goes to the session log, so it cannot be confused with a reply. Every
  reply names the worker and the request it answers; replies for other
  requests are ignored.
- A `generate` request only names a `job.json` that must be inside the job
  root; the worker validates it like the one-shot runner and checks that its
  runtime settings equal the worker's. A second request while busy, or a
  reused request id, is refused.
- Only one worker runs at a time per ComfyUI process, and jobs are
  serialised. A job is never re-sent after the worker accepted it.
- The worker exits on its own when ComfyUI closes the pipe or dies (idle or
  busy; it then kills its own process group), after `worker_idle_seconds`
  without a job (default 300), on Unload, or after a CUDA error. ComfyUI also
  unloads it when it exits normally.
- Cancel and timeout first ask the worker to stop between generator forwards
  (it stays loaded). If that is not confirmed within 60 s, the worker is
  stopped with the same kill helper as one-shot jobs, limited to its own
  process group, and the next job starts a new worker.

## Outputs

`result.json` is size-limited, must name this job, and each video path must
match `videos/NN.mp4` inside the job directory with the recorded sha256 and
the expected 81 frames at 832x480. The node returns ComfyUI `VIDEO` objects
for those files.

## Reporting a vulnerability

Please open a GitHub security advisory on this repository (Security tab ->
Report a vulnerability) instead of a public issue.
