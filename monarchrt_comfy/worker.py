"""Persistent worker backend: keep one warm MonarchRT runtime process between ComfyUI queue jobs.

The ComfyUI process owns a single WorkerManager. It starts ``runtime/monarchrt_worker.py`` for an
administrator-registered runtime (same argv/environment rules as the one-shot backend, see client.py),
talks to it over the inherited stdin/stdout pipes (JSON lines, no network port) and keeps it until:

- a job needs a different runtime or the runtime code/config changed (identity mismatch) -> restart,
- the worker has been idle for ``worker_idle_seconds`` -> it exits by itself,
- ComfyUI exits or dies -> the worker sees EOF on stdin and stops its own process group,
- the Unload node / ``unload()`` is called, a job was cancelled without a clean stop, or the worker
  reported a fatal (CUDA) error.

At most one worker runs at a time and jobs are serialised with a lock. A job is never retried after the
worker accepted it, so a video can not be produced twice.
"""

from __future__ import annotations

import atexit
import hashlib
import json
import os
import queue
import shutil
import subprocess
import threading
import time
import uuid
from collections import deque
from collections.abc import Callable
from pathlib import Path

from . import client
from .config import Runtime

PROTOCOL_VERSION = 1
WORKER_SCRIPT = "monarchrt_worker.py"
WORKER_ENTRY = WORKER_SCRIPT
# Files copied into the session folder at start-up: the worker runs this frozen copy, so updating the
# node while a worker is alive never changes the code it executes (the identity changes -> restart).
CODE_FILES = {WORKER_SCRIPT: client.RUNTIME_DIR / WORKER_SCRIPT, client.JOB_SCRIPT: client.RUNTIME_DIR / client.JOB_SCRIPT}
CANCEL_GRACE_S = 60.0
STATUS_TIMEOUT_S = 10.0
MAX_MESSAGE_BYTES = 1024 * 1024


class WorkerError(client.RuntimeJobError):
    pass


def code_digest() -> str:
    h = hashlib.sha256()
    for name in sorted(CODE_FILES):
        h.update(name.encode())
        h.update(Path(CODE_FILES[name]).read_bytes())
    return h.hexdigest()


def identity(rt: Runtime) -> str:
    """Everything that decides which model and code a worker has loaded (not the attention profile)."""
    data = {
        "protocol": PROTOCOL_VERSION,
        "runtime_id": rt.id,
        "kind": rt.kind,
        "distro": rt.distro,
        "python": rt.python,
        "upstream_dir": rt.upstream_dir,
        "models_dir": rt.models_dir,
        "checkpoint": rt.checkpoint,
        "env": dict(sorted(rt.env.items())),
        "offload_text_encoder": rt.offload_text_encoder,
        "code": code_digest(),
    }
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()


class WorkerProcess:
    def __init__(self, rt: Runtime, ident: str):
        self.rt = rt
        self.identity = ident
        self.worker_id = f"worker-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:12]}"
        root = client.jobs_root(rt)
        self.session_dir = (root / self.worker_id).resolve()
        if self.session_dir.parent != root:
            raise WorkerError("worker session escaped the jobs root")
        self.messages: queue.Queue = queue.Queue()
        self.proc: subprocess.Popen | None = None
        self.ready: dict | None = None
        self.bye: dict | None = None
        self.started_at = time.time()
        self.jobs = 0
        self.stop_report: dict | None = None

    # -- process ---------------------------------------------------------------------------------------
    def start(self) -> None:
        code = self.session_dir / "code"
        code.mkdir(parents=True)
        for name, src in CODE_FILES.items():
            shutil.copyfile(src, code / name)
        cfg = {
            "schema_version": 1,
            "worker_id": self.worker_id,
            "runtime_id": self.rt.id,
            "identity": self.identity,
            "upstream_dir": self.rt.upstream_dir,
            "models_dir": self.rt.models_dir,
            "checkpoint": self.rt.checkpoint,
            "env": dict(self.rt.env),
            "offload_text_encoder": self.rt.offload_text_encoder,
            "idle_seconds": self.rt.worker_idle_seconds,
            "jobs_root": client.host_to_runtime_path(self.rt, client.jobs_root(self.rt)),
        }
        (self.session_dir / "worker.json").write_text(json.dumps(cfg, indent=1), encoding="utf-8")
        argv = client.argv_for(
            self.rt,
            client.host_to_runtime_path(self.rt, code / WORKER_ENTRY),
            client.host_to_runtime_path(self.rt, self.session_dir / "worker.json"),
        )
        kwargs: dict = {}
        if os.name == "nt":
            kwargs["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0) | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        else:
            kwargs["start_new_session"] = True
        with open(self.session_dir / "worker.log", "wb") as log:
            self.proc = subprocess.Popen(  # noqa: S603 - argv from admin config + frozen package scripts, no shell
                argv,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=log,
                env=client.child_env(self.rt),
                cwd=str(self.session_dir),
                close_fds=True,
                **kwargs,
            )
        threading.Thread(target=self._read, name=f"{self.worker_id}-reader", daemon=True).start()

    def _read(self) -> None:
        assert self.proc is not None and self.proc.stdout is not None
        for raw in self.proc.stdout:
            if len(raw) > MAX_MESSAGE_BYTES:
                continue
            try:
                msg = json.loads(raw.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                continue  # the worker writes only protocol lines here; ignore anything else
            if not isinstance(msg, dict) or msg.get("worker_id") != self.worker_id:
                continue  # not from this worker: never mix up replies
            if msg.get("type") == "bye":
                self.bye = msg
            self.messages.put(msg)
        self.messages.put({"type": "_closed"})

    def alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def send(self, msg_type: str, **fields) -> str:
        rid = fields.pop("request_id", None) or f"req-{uuid.uuid4().hex}"
        line = json.dumps({"v": PROTOCOL_VERSION, "type": msg_type, "request_id": rid, **fields}) + "\n"
        if self.proc is None or self.proc.stdin is None:
            raise WorkerError("worker is not running")
        try:
            self.proc.stdin.write(line.encode("utf-8"))
            self.proc.stdin.flush()
        except OSError as exc:
            raise WorkerError("worker pipe is closed") from exc
        return rid

    def wait_ready(self, timeout_s: float, interrupted: Callable[[], bool]) -> dict:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            try:
                msg = self.messages.get(timeout=0.25)
            except queue.Empty:
                if interrupted():
                    raise client.JobCancelled("cancelled while the worker was starting") from None
                if not self.alive():
                    raise WorkerError(f"worker exited during start-up (exit {self.proc.poll()}): {self.log_tail()}") from None
                continue
            if msg.get("type") == "ready":
                if msg.get("identity") != self.identity:
                    raise WorkerError("worker reported a different identity than requested")
                self.ready = msg
                return msg
            if msg.get("type") == "error" and msg.get("fatal"):
                raise WorkerError(f"worker start-up failed: {msg.get('error_type')}: {msg.get('error')}")
            if msg.get("type") == "_closed":
                raise WorkerError(f"worker exited during start-up: {self.log_tail()}")
        raise client.JobTimeout(f"worker did not become ready within {timeout_s:.0f} s")

    def log_tail(self, n: int = 1500) -> str:
        try:
            return (self.session_dir / "worker.log").read_bytes()[-n:].decode("utf-8", "replace")
        except OSError:
            return ""

    def stop(self, reason: str, grace_s: float = 15.0) -> dict:
        """Ask the worker to exit, then make sure its whole process group is gone."""
        if self.proc is not None and self.proc.poll() is None:
            try:
                self.send("shutdown")
            except WorkerError:
                pass
        if self.proc is not None and self.proc.stdin is not None:
            try:
                self.proc.stdin.close()
            except OSError:
                pass
        if self.proc is not None:
            try:
                self.proc.wait(timeout=grace_s)
            except subprocess.TimeoutExpired:
                pass
        report = client.stop_job(self.rt, self.proc, self.session_dir, grace_s=0)
        report["reason"] = reason
        self.stop_report = report
        return report


class WorkerManager:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._worker: WorkerProcess | None = None
        self.events: deque = deque(maxlen=50)  # lifecycle log for status/doctor (no prompts, no paths)

    def _note(self, event: str, **fields) -> None:
        self.events.append({"t": round(time.time(), 1), "event": event, **fields})

    def _discard(self, reason: str) -> None:
        w = self._worker
        self._worker = None
        if w is not None:
            report = w.stop(reason)
            self._note("worker_stopped", worker_id=w.worker_id, reason=reason, jobs=w.jobs, stop=report.get("status"))

    def _ensure(self, rt: Runtime, interrupted: Callable[[], bool]) -> WorkerProcess:
        ident = identity(rt)
        w = self._worker
        if w is not None:
            if not w.alive():
                why = (w.bye or {}).get("reason", f"exit code {w.proc.poll() if w.proc else None}")
                self._note("worker_gone", worker_id=w.worker_id, reason=why)
                self._discard(f"worker had exited ({why})")
            elif w.identity != ident:
                self._discard("runtime, code or settings changed")
            else:
                return w
        w = WorkerProcess(rt, ident)
        self._note("worker_starting", worker_id=w.worker_id, runtime_id=rt.id)
        w.start()
        self._worker = w
        try:
            ready = w.wait_ready(rt.timeout_minutes * 60, interrupted)
        except BaseException as exc:
            self._discard(f"start-up failed: {type(exc).__name__}")
            raise
        self._note("worker_ready", worker_id=w.worker_id, pid=ready.get("pid"), load_s=ready.get("model_load_seconds"))
        return w

    def generate(
        self,
        rt: Runtime,
        profile: str,
        videos: list[dict],
        *,
        interrupted: Callable[[], bool] = lambda: False,
        on_progress: Callable[[int, int], None] | None = None,
        timeout_s: float | None = None,
    ) -> client.JobOutcome:
        with self._lock:
            t0 = time.time()
            job_id, job_dir = client.new_job_dir(rt, "job")
            job = client.make_job(rt, job_id, profile, videos)
            client._write_json(job_dir / "job.json", job)
            total = client.FORWARDS_PER_VIDEO * len(videos)
            deadline = time.monotonic() + (timeout_s or rt.timeout_minutes * 60)
            before = self._worker.worker_id if self._worker is not None and self._worker.alive() else None
            for attempt in (1, 2):
                w = self._ensure(rt, interrupted)
                started_new = w.worker_id != before
                rid = f"req-{uuid.uuid4().hex}"
                try:
                    w.send("generate", request_id=rid, job=client.host_to_runtime_path(rt, job_dir / "job.json"))
                except WorkerError:
                    self._discard("pipe closed before the job was sent")
                    if attempt == 1:
                        continue
                    raise
                try:
                    outcome = self._follow(w, rid, job, job_dir, total, on_progress, interrupted, deadline)
                except (client.JobCancelled, client.JobTimeout, WorkerError):
                    raise  # already handled: the worker was told to stop the job, or was replaced
                except BaseException:
                    # anything else leaving this loop (e.g. ComfyUI's progress hook raising its interrupt
                    # exception from on_progress) must not leave the job running in the worker
                    if self._worker is w and w.alive():
                        self._cancel(w, rid)
                    raise
                if outcome == "resend" and attempt == 1:
                    before = None
                    continue  # the worker ended (e.g. idle timeout) before accepting the job: safe to send once more
                if isinstance(outcome, client.JobOutcome):
                    w.jobs += 1
                    outcome.result.setdefault("worker", {})["manager_wall_seconds"] = round(time.time() - t0, 3)
                    outcome.result["worker"]["started_for_this_job"] = started_new
                    self._note("job_done", worker_id=w.worker_id, job_id=job_id, profile=profile, videos=len(videos), wall_s=round(time.time() - t0, 2))
                    return outcome
                raise WorkerError("worker exited before accepting the job twice")
            raise WorkerError("unreachable")

    def _follow(self, w, rid, job, job_dir, total, on_progress, interrupted, deadline):
        accepted = False
        while True:
            try:
                msg = w.messages.get(timeout=0.25)
            except queue.Empty:
                msg = None
            # every iteration also reaches the liveness / interrupt / deadline checks below, even while
            # progress messages keep arriving
            if msg is not None:
                kind = msg.get("type")
                if kind == "_closed" or (kind == "bye" and msg.get("request_id") is None):
                    why = msg.get("reason") or f"exit code {w.proc.poll() if w.proc else None}"
                    self._discard(f"worker ended during a job ({why})")
                    if not accepted:
                        return "resend"
                    raise WorkerError(f"the worker exited during the job ({why}); the next job starts a new worker")
                if msg.get("request_id") == rid:  # anything else is a stale or unrelated reply
                    if kind == "accepted":
                        accepted = True
                    elif kind == "progress" and on_progress is not None:
                        done = int(msg.get("video", 0)) * client.FORWARDS_PER_VIDEO + int(msg.get("forwards", 0))
                        on_progress(min(done, total), total)
                    elif kind == "result":
                        return client.load_result(job_dir, job)
                    elif kind == "cancelled":
                        raise client.JobCancelled("the job was cancelled")
                    elif kind == "error":
                        if msg.get("fatal"):
                            self._discard(f"fatal worker error: {msg.get('error_type')}")
                        raise WorkerError(f"runtime job failed: {msg.get('error_type')}: {msg.get('error')}")
            if msg is None and not w.alive():
                self._discard(f"worker process ended (exit {w.proc.poll()})")
                if not accepted:
                    return "resend"
                raise WorkerError(f"the worker exited during the job: {w.log_tail()}")
            if interrupted():
                self._cancel(w, rid)
                raise client.JobCancelled("cancelled by the user")
            if time.monotonic() > deadline:
                self._cancel(w, rid)
                raise client.JobTimeout("runtime job exceeded its time limit")

    def _cancel(self, w: WorkerProcess, rid: str) -> None:
        """Cooperative cancel (between forwards); if the worker does not confirm in time, stop it."""
        try:
            w.send("cancel", target=rid)
        except WorkerError:
            self._discard("pipe closed during cancel")
            return
        deadline = time.monotonic() + CANCEL_GRACE_S
        while time.monotonic() < deadline:
            try:
                msg = w.messages.get(timeout=0.25)
            except queue.Empty:
                if not w.alive():
                    break
                continue
            if msg.get("request_id") == rid and msg.get("type") in ("cancelled", "result", "error"):
                self._note("job_cancelled", worker_id=w.worker_id, clean=msg.get("type") == "cancelled")
                if msg.get("type") == "error" and msg.get("fatal"):
                    self._discard("fatal error during cancel")
                return
            if msg.get("type") == "_closed":
                break
        self._discard("cancel not confirmed in time")

    def status(self) -> dict:
        with self._lock:
            w = self._worker
            out: dict = {"backend": "persistent", "events": list(self.events)[-12:]}
            if w is None:
                out["worker"] = None
                return out
            out["worker"] = {
                "worker_id": w.worker_id,
                "runtime_id": w.rt.id,
                "alive": w.alive(),
                "jobs_via_this_manager": w.jobs,
                "started_s_ago": round(time.time() - w.started_at, 1),
                "ready": {k: (w.ready or {}).get(k) for k in ("pid", "pgid", "startup_seconds", "model_load_seconds")},
                "bye": w.bye,
            }
            if w.alive():
                try:
                    rid = w.send("status")
                    end = time.monotonic() + STATUS_TIMEOUT_S
                    while time.monotonic() < end:
                        try:
                            msg = w.messages.get(timeout=0.25)
                        except queue.Empty:
                            continue
                        if msg.get("request_id") == rid and msg.get("type") == "status":
                            out["worker"]["live"] = {k: v for k, v in msg.items() if k not in ("v", "type", "request_id")}
                            break
                        if msg.get("type") == "_closed":  # keep the end-of-stream marker for the next job
                            w.messages.put(msg)
                            break
                except WorkerError as exc:
                    out["worker"]["status_error"] = str(exc)
            return out

    def unload(self, reason: str = "unload requested") -> dict:
        with self._lock:
            w = self._worker
            if w is None:
                return {"unloaded": False, "reason": "no worker running"}
            self._discard(reason)
            return {"unloaded": True, "worker_id": w.worker_id, "stop": w.stop_report}


MANAGER = WorkerManager()


@atexit.register
def _shutdown_on_exit() -> None:  # ComfyUI exits normally: do not leave a warm worker behind
    try:
        MANAGER.unload("ComfyUI is exiting")
    except Exception:
        pass
