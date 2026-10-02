"""Persistent MonarchRT worker: load the pipeline once, then serve jobs over stdin/stdout (runtime side).

Usage: monarchrt_worker.py <session_dir>/worker.json

ComfyUI (monarchrt_comfy/worker.py) starts one worker per runtime, keeps the pipe open and sends one
request per line; this process answers on a private copy of the original stdout. Everything the upstream
code prints goes to stderr (the session log), so it can never be mistaken for a protocol message.

Requests (JSON, one per line, at most 64 KiB):
  {"v": 1, "type": "generate", "request_id": ID, "job": "<path of job.json inside jobs_root>"}
  {"v": 1, "type": "cancel",   "request_id": ID, "target": ID}
  {"v": 1, "type": "status",   "request_id": ID}
  {"v": 1, "type": "shutdown", "request_id": ID}
Replies carry "worker_id" and the "request_id" they answer:
  ready, accepted, progress, result, cancelled, error, status, bye

Lifetime: the worker exits when stdin reaches EOF (ComfyUI closed the pipe or died, idle or busy), on
"shutdown", or after `idle_seconds` without a request. On every exit it kills the rest of its own
process group (compile workers etc.). Jobs run one at a time; a second "generate" while busy, or a
reused request id, is answered with an error. Per-video state (seed, noise, KV / cross-attention caches,
VAE cache, counters, timers) is reset by Engine.generate for every video; nothing of a previous job is
reused except the loaded weights, the patched modules and Triton's compiled / tuned kernels.

Exit codes: 0 normal, 2 invalid configuration, 3 startup failure.
"""

from __future__ import annotations

import json
import os
import queue
import re
import sys
import threading
import time
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import monarchrt_job as job_mod  # noqa: E402

PROTOCOL_VERSION = 1
MAX_LINE = 64 * 1024
_ID_RE = re.compile(r"[A-Za-z0-9_-]{8,80}")
_CONFIG_KEYS = {
    "schema_version",
    "worker_id",
    "runtime_id",
    "identity",
    "upstream_dir",
    "models_dir",
    "checkpoint",
    "env",
    "offload_text_encoder",
    "idle_seconds",
    "jobs_root",
    "initial_profile",
}


class ConfigError(ValueError):
    pass


def load_config(path: Path) -> dict:
    if path.stat().st_size > 64 * 1024:
        raise ConfigError("worker.json too large")
    cfg = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(cfg, dict) or cfg.get("schema_version") != 1 or set(cfg) - _CONFIG_KEYS:
        raise ConfigError("unsupported worker.json")
    if not isinstance(cfg.get("worker_id"), str) or not _ID_RE.fullmatch(cfg["worker_id"]):
        raise ConfigError("invalid worker_id")
    for key in ("upstream_dir", "models_dir", "checkpoint", "jobs_root"):
        if not isinstance(cfg.get(key), str) or not os.path.isabs(cfg[key]):
            raise ConfigError(f"{key} must be an absolute path")
    env = cfg.get("env", {})
    if not isinstance(env, dict) or set(env) - job_mod.ENV_KEYS or not all(isinstance(v, str) for v in env.values()):
        raise ConfigError("env not allowed")
    idle = cfg.get("idle_seconds", 300)
    if not isinstance(idle, int) or isinstance(idle, bool) or not 10 <= idle <= 24 * 3600:
        raise ConfigError("idle_seconds must be an integer in 10..86400")
    if cfg.get("initial_profile", "monarch_h2") not in job_mod.PROFILES:
        raise ConfigError("unknown initial_profile")
    if not isinstance(cfg.get("offload_text_encoder", False), bool):
        raise ConfigError("offload_text_encoder must be a boolean")
    return cfg


class Proto:
    """Line-oriented JSON writer on a private fd (thread safe)."""

    def __init__(self, fd: int, worker_id: str):
        self._f = os.fdopen(fd, "w", buffering=1, encoding="utf-8")
        self._lock = threading.Lock()
        self.worker_id = worker_id

    def send(self, msg_type: str, **fields) -> None:
        line = json.dumps({"v": PROTOCOL_VERSION, "type": msg_type, "worker_id": self.worker_id, "t": round(time.time(), 3), **fields}, ensure_ascii=False)
        with self._lock:
            try:
                self._f.write(line + "\n")
                self._f.flush()
            except (BrokenPipeError, OSError):
                pass


def _rss_bytes() -> int | None:
    try:
        with open("/proc/self/status", encoding="ascii") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        return None
    return None


class Worker:
    def __init__(self, cfg: dict, proto: Proto, engine_factory):
        self.cfg = cfg
        self.proto = proto
        self.engine_factory = engine_factory
        self.engine = None
        self.jobs_root = os.path.realpath(cfg["jobs_root"])
        self.requests: queue.Queue = queue.Queue()
        self.lock = threading.Lock()
        self.busy_with: str | None = None
        self.cancel_target: str | None = None
        self.seen_ids: set[str] = set()
        self.started = time.time()
        self.last_activity = time.time()
        self.jobs_done = 0
        self.jobs_failed = 0
        self.jobs_cancelled = 0
        self.history: list[dict] = []  # last few jobs: request id, job id, profile, seconds (no prompts)

    # -- reader thread: validates requests, answers status/cancel at once, queues the rest ---------------
    def reader(self) -> None:
        buf = b""
        while True:
            try:
                chunk = os.read(0, 65536)
            except OSError:
                chunk = b""
            if not chunk:
                # the parent closed the pipe or died: never keep running on our own, idle or busy
                if self.busy_with is not None:
                    self.proto.send("bye", request_id=None, reason="stdin closed while busy", jobs_done=self.jobs_done)
                    sys.stderr.flush()
                    job_mod._stop_own_group()
                    os._exit(0)
                self.requests.put({"type": "_eof"})
                return
            buf += chunk
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                self._handle_line(line)
            if len(buf) > MAX_LINE:
                buf = b""
                self.proto.send("error", request_id=None, error_type="ProtocolError", error="request line too long", fatal=False)

    def _handle_line(self, line: bytes) -> None:
        if not line.strip():
            return
        try:
            msg = json.loads(line.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            self.proto.send("error", request_id=None, error_type="ProtocolError", error="not UTF-8 JSON", fatal=False)
            return
        rid = msg.get("request_id") if isinstance(msg, dict) else None
        if not isinstance(msg, dict) or msg.get("v") != PROTOCOL_VERSION or not isinstance(rid, str) or not _ID_RE.fullmatch(rid):
            self.proto.send("error", request_id=rid if isinstance(rid, str) else None, error_type="ProtocolError", error="bad request envelope", fatal=False)
            return
        kind = msg.get("type")
        with self.lock:
            if rid in self.seen_ids:
                self.proto.send("error", request_id=rid, error_type="DuplicateRequest", error="request_id already used", fatal=False)
                return
            self.seen_ids.add(rid)
            if kind == "status":
                self.proto.send("status", request_id=rid, **self._status())
                return
            if kind == "cancel":
                target = msg.get("target")
                if target is not None and target == self.busy_with:
                    self.cancel_target = target
                    self.proto.send("accepted", request_id=rid, target=target)
                else:
                    self.proto.send("error", request_id=rid, error_type="NotRunning", error="no such running request", fatal=False)
                return
            if kind == "generate":
                if self.busy_with is not None or not self.requests.empty():
                    self.proto.send("error", request_id=rid, error_type="Busy", error="worker is busy; jobs run one at a time", fatal=False)
                    return
                if not isinstance(msg.get("job"), str):
                    self.proto.send("error", request_id=rid, error_type="ProtocolError", error="generate needs a job path", fatal=False)
                    return
                self.busy_with = rid  # reserve before queueing so a second generate is refused
                self.requests.put(msg)
                return
            if kind == "shutdown":
                self.requests.put(msg)
                return
        self.proto.send("error", request_id=rid, error_type="ProtocolError", error=f"unknown request type {kind!r}", fatal=False)

    def _status(self) -> dict:
        st = {
            "pid": os.getpid(),
            "pgid": os.getpgid(0) if hasattr(os, "getpgid") else None,
            "uptime_s": round(time.time() - self.started, 1),
            "idle_s": round(time.time() - self.last_activity, 1) if self.busy_with is None else 0.0,
            "idle_limit_s": self.cfg.get("idle_seconds", 300),
            "busy_with": self.busy_with,
            "engine_loaded": self.engine is not None,
            "model_loads": 1 if self.engine is not None else 0,
            "jobs_done": self.jobs_done,
            "jobs_failed": self.jobs_failed,
            "jobs_cancelled": self.jobs_cancelled,
            "autotune_bench_calls_total": job_mod._AUTOTUNE["bench_calls"],
            "rss_bytes": _rss_bytes(),
            "history": self.history[-8:],
        }
        if self.engine is not None:
            st["profile"] = self.engine.profile
            st["videos_generated"] = self.engine.videos_generated
            try:
                import torch

                st["cuda_allocated_bytes"] = torch.cuda.memory_allocated()
                st["cuda_reserved_bytes"] = torch.cuda.memory_reserved()
            except Exception:
                pass
        return st

    # -- main loop -----------------------------------------------------------------------------------------
    def load(self) -> None:
        t0 = time.time()
        self.engine = self.engine_factory(
            Path(self.cfg["upstream_dir"]),
            Path(self.cfg["models_dir"]),
            Path(self.cfg["checkpoint"]),
            self.cfg.get("initial_profile", "monarch_h2"),
            bool(self.cfg.get("offload_text_encoder", False)),
        )
        self.engine.should_cancel = lambda: self.cancel_target is not None and self.cancel_target == self.busy_with
        self.proto.send(
            "ready",
            request_id=None,
            identity=self.cfg.get("identity"),
            runtime_id=self.cfg.get("runtime_id"),
            pid=os.getpid(),
            pgid=os.getpgid(0) if hasattr(os, "getpgid") else None,
            startup_seconds=round(time.time() - t0, 3),
            model_load_seconds=round(self.engine.load_seconds, 3),
            weights=self.engine.load_report,
            versions=self.engine.versions(),
        )
        self.last_activity = time.time()

    def serve(self) -> str:
        idle_limit = self.cfg.get("idle_seconds", 300)
        while True:
            try:
                msg = self.requests.get(timeout=1.0)
            except queue.Empty:
                if self.busy_with is None and time.time() - self.last_activity > idle_limit:
                    return "idle timeout"
                continue
            kind = msg.get("type")
            if kind == "_eof":
                return "stdin closed"
            if kind == "shutdown":
                return "shutdown requested"
            if kind == "generate":
                self._generate(msg)
                self.last_activity = time.time()

    def _resolve_job(self, raw: str) -> Path:
        path = os.path.realpath(raw)
        if not path.startswith(self.jobs_root + os.sep) or os.path.basename(path) != "job.json":
            raise job_mod.JobError("job path outside the jobs root")
        return Path(path)

    def _generate(self, msg: dict) -> None:
        rid = msg["request_id"]
        t0 = time.time()
        job_id = None
        try:
            job_path = self._resolve_job(msg["job"])
            job = job_mod.load_job(job_path)
            job_id = job["job_id"]
            for key in ("upstream_dir", "models_dir", "checkpoint"):
                if os.path.realpath(job[key]) != os.path.realpath(self.cfg[key]):
                    raise job_mod.JobError(f"job {key} does not match this worker")
            if bool(job.get("offload_text_encoder", False)) != bool(self.cfg.get("offload_text_encoder", False)) or job.get("env", {}) != self.cfg.get(
                "env", {}
            ):
                raise job_mod.JobError("job runtime settings do not match this worker")
            job_dir = job_path.parent
            out_dir = job_dir / "videos"
            if out_dir.exists():
                raise job_mod.JobError("job already has outputs")
            self.proto.send("accepted", request_id=rid, job_id=job_id)  # from here on the job is never re-sent
            out_dir.mkdir()
            effective = self.engine.set_profile(job["profile"])
            bench0 = job_mod._AUTOTUNE["bench_calls"]
            self.engine.on_progress = lambda video, forwards: self.proto.send(
                "progress", request_id=rid, video=video, forwards=forwards, forwards_per_video=job_mod.FORWARDS_PER_VIDEO
            )
            records = []
            for idx, video in enumerate(job["videos"]):
                records.append(self.engine.generate(idx, video, out_dir, job.get("fps", 16)))
            self.jobs_done += 1
            job_mod.write_result(
                job_dir,
                {
                    "schema_version": job_mod.SCHEMA_VERSION,
                    "job_id": job_id,
                    "profile": job["profile"],
                    "backend": "persistent",
                    "worker": {
                        "worker_id": self.proto.worker_id,
                        "pid": os.getpid(),
                        "request_id": rid,
                        "job_number_in_worker": self.jobs_done,
                        "model_loads_in_worker": 1,
                        "worker_model_load_seconds": round(self.engine.load_seconds, 3),
                        "autotune_bench_calls_this_job": job_mod._AUTOTUNE["bench_calls"] - bench0,
                        "rss_bytes_after": _rss_bytes(),
                    },
                    "effective_config": effective,
                    "weights": self.engine.load_report,
                    "versions": self.engine.versions(),
                    "model_load_seconds": 0.0,  # loaded before this job (see worker.worker_model_load_seconds)
                    "process_seconds": round(time.time() - t0, 3),
                    "host_peak_rss_bytes": job_mod._peak_rss_bytes(),
                    "videos": records,
                },
            )
            self.history.append({"request_id": rid, "job_id": job_id, "profile": job["profile"], "videos": len(records), "seconds": round(time.time() - t0, 2)})
            self.proto.send("result", request_id=rid, job_id=job_id, result="result.json")
        except job_mod.JobCancelled:
            self.jobs_cancelled += 1
            self.proto.send("cancelled", request_id=rid, job_id=job_id)
        except Exception as exc:
            import traceback

            self.jobs_failed += 1
            traceback.print_exc(file=sys.stderr)
            # CUDA errors can leave the context unusable: tell the parent to replace this worker
            fatal = "CUDA" in str(exc) or "cuda" in type(exc).__module__.lower()
            self.proto.send("error", request_id=rid, job_id=job_id, error_type=type(exc).__name__, error=str(exc)[:2000], fatal=fatal)
            if fatal:
                raise
        finally:
            with self.lock:
                self.busy_with = None
                self.cancel_target = None


def main(argv: list[str], engine_factory=None) -> int:
    if len(argv) != 2:
        print("usage: monarchrt_worker.py <worker.json>", file=sys.stderr)
        return 2
    cfg_path = Path(argv[1]).resolve()
    try:
        cfg = load_config(cfg_path)
    except (ConfigError, OSError, json.JSONDecodeError) as exc:
        print(f"invalid worker configuration: {exc}", file=sys.stderr)
        return 2
    job_mod.apply_env(cfg.get("env", {}))
    job_mod.become_session_leader(cfg_path.parent / "runner.pid")
    # Protocol on a private copy of stdout; fd 1 and sys.stdout now go to stderr (the session log).
    proto_fd = os.dup(1)
    os.dup2(2, 1)
    sys.stdout = sys.stderr
    proto = Proto(proto_fd, cfg["worker_id"])
    worker = Worker(cfg, proto, engine_factory or job_mod.Engine)
    threading.Thread(target=worker.reader, daemon=True).start()
    try:
        worker.load()
    except Exception as exc:
        import traceback

        traceback.print_exc(file=sys.stderr)
        proto.send("error", request_id=None, error_type=type(exc).__name__, error=str(exc)[:2000], fatal=True, phase="startup")
        return 3
    try:
        reason = worker.serve()
    except Exception as exc:
        reason = f"fatal error: {type(exc).__name__}"
    proto.send("bye", request_id=None, reason=reason, jobs_done=worker.jobs_done)
    return 0


if __name__ == "__main__":
    code = main(sys.argv)
    sys.stderr.flush()
    job_mod._stop_own_group()  # compile workers and other helpers never outlive the worker
    os._exit(code)
