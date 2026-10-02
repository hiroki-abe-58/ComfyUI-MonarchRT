"""Persistent worker: real subprocesses running the real worker code with a fake engine (no GPU)."""

from __future__ import annotations

import json
import os
import queue
import subprocess
import sys
import textwrap
import threading
import time
from pathlib import Path

import pytest

from monarchrt_comfy import client
from monarchrt_comfy import worker as wmod
from monarchrt_comfy.config import parse_runtime

POSIX = os.name == "posix"
FAKE = Path(__file__).resolve().parent / "fake_runtime"


@pytest.fixture(scope="session")
def fixture_mp4(tmp_path_factory):
    import av
    import numpy as np

    path = tmp_path_factory.mktemp("fixture") / "fixture.mp4"
    with av.open(str(path), "w") as c:
        s = c.add_stream("libx264", rate=16)
        s.width, s.height, s.pix_fmt = 832, 480, "yuv420p"
        s.options = {"crf": "35", "preset": "ultrafast"}
        for i in range(81):
            for p in s.encode(av.VideoFrame.from_ndarray(np.full((480, 832, 3), i * 3 % 256, dtype=np.uint8), format="rgb24")):
                c.mux(p)
        for p in s.encode():
            c.mux(p)
    return path


def make_rt(tmp_path, **over):
    raw = {
        "kind": "posix",
        "python": "/fake/python",
        "upstream_dir": "/fake/upstream",
        "models_dir": "/fake/models",
        "checkpoint": "/fake/ckpt.pt",
        "jobs_dir": str(tmp_path / "jobs"),
        "timeout_minutes": 5,
        "backend": "persistent",
        "worker_idle_seconds": 300,
    }
    raw.update(over)
    rt = parse_runtime("fake", raw)
    object.__setattr__(rt, "python", sys.executable)  # a host path; real 'posix' configs require Linux paths
    return rt


@pytest.fixture
def mgr(monkeypatch, fixture_mp4):
    files = dict(wmod.CODE_FILES)
    files["fake_monarchrt_worker.py"] = FAKE / "fake_monarchrt_worker.py"
    files["fixture.mp4"] = fixture_mp4
    monkeypatch.setattr(wmod, "CODE_FILES", files)
    monkeypatch.setattr(wmod, "WORKER_ENTRY", "fake_monarchrt_worker.py")
    m = wmod.WorkerManager()
    yield m
    m.unload("test teardown")


def _alive(pid: int) -> bool:
    if POSIX:
        try:
            with open(f"/proc/{pid}/stat", "rb") as f:
                return f.read().rsplit(b")", 1)[1].split()[0] != b"Z"
        except OSError:
            return False
    import ctypes

    h = ctypes.windll.kernel32.OpenProcess(0x1000 | 0x00100000, False, pid)
    if not h:
        return False
    try:
        return ctypes.windll.kernel32.WaitForSingleObject(h, 0) == 0x102
    finally:
        ctypes.windll.kernel32.CloseHandle(h)


def _wait_gone(pid: int, seconds: float = 20) -> bool:
    end = time.time() + seconds
    while time.time() < end:
        if not _alive(pid):
            return True
        time.sleep(0.1)
    return False


def gen(m, rt, prompt="MODE:ok a fox", seed=1, profile="monarch_h2", **kw):
    return m.generate(rt, profile, [{"prompt": prompt, "seed": seed}], **kw)


def test_three_queue_jobs_reuse_one_worker(mgr, tmp_path):
    rt = make_rt(tmp_path)
    outs = [gen(mgr, rt, seed=s) for s in (1, 2, 1)]
    pids = {o.result["videos"][0]["worker_pid"] for o in outs}
    engines = {o.result["videos"][0]["engine_id"] for o in outs}
    assert len(pids) == 1 and len(engines) == 1, "jobs did not share the worker"
    assert [o.result["videos"][0]["engine_loads"] for o in outs] == [1, 1, 1]
    assert [o.result["worker"]["started_for_this_job"] for o in outs] == [True, False, False]
    assert [o.result["worker"]["job_number_in_worker"] for o in outs] == [1, 2, 3]
    assert len({o.job_dir for o in outs}) == 3  # every job has its own folder and its own result
    assert all(o.result["backend"] == "persistent" for o in outs)


def test_profile_switch_keeps_worker(mgr, tmp_path):
    rt = make_rt(tmp_path)
    outs = [gen(mgr, rt, profile=p) for p in ("dense", "monarch_h2", "dense")]
    assert len({o.result["videos"][0]["worker_pid"] for o in outs}) == 1
    assert [o.result["videos"][0]["attention_dispatch"]["profile"] for o in outs] == ["dense", "monarch_h2", "dense"]


def test_identity_change_restarts_and_stops_old_worker(mgr, tmp_path):
    a = gen(mgr, make_rt(tmp_path))
    b = gen(mgr, make_rt(tmp_path, offload_text_encoder=True))
    pa, pb = a.result["videos"][0]["worker_pid"], b.result["videos"][0]["worker_pid"]
    assert pa != pb and b.result["worker"]["started_for_this_job"] is True
    assert _wait_gone(pa), "old worker still running"


def test_status_and_unload(mgr, tmp_path):
    rt = make_rt(tmp_path)
    out = gen(mgr, rt)
    pid = out.result["videos"][0]["worker_pid"]
    st = mgr.status()
    assert st["worker"]["alive"] and st["worker"]["live"]["jobs_done"] == 1 and st["worker"]["live"]["model_loads"] == 1
    assert mgr.unload()["unloaded"] is True
    assert _wait_gone(pid)
    assert mgr.status()["worker"] is None
    again = gen(mgr, rt)
    assert again.result["worker"]["started_for_this_job"] is True and again.result["videos"][0]["worker_pid"] != pid


def test_idle_timeout_exits_and_next_job_restarts(mgr, tmp_path):
    rt = make_rt(tmp_path, worker_idle_seconds=10)
    pid = gen(mgr, rt).result["videos"][0]["worker_pid"]
    assert _wait_gone(pid, 25), "worker did not exit after the idle timeout"
    assert mgr._worker.bye["reason"] == "idle timeout"
    nxt = gen(mgr, rt)
    assert nxt.result["worker"]["started_for_this_job"] is True


def test_cooperative_cancel_keeps_worker_warm(mgr, tmp_path):
    rt = make_rt(tmp_path)
    pid = gen(mgr, rt).result["videos"][0]["worker_pid"]
    t0 = time.time()
    with pytest.raises(client.JobCancelled):
        gen(mgr, rt, prompt="MODE:hang", interrupted=lambda: time.time() - t0 > 1.5)
    nxt = gen(mgr, rt)
    assert nxt.result["videos"][0]["worker_pid"] == pid and nxt.result["worker"]["job_number_in_worker"] == 2
    assert any(e["event"] == "job_cancelled" and e["clean"] for e in mgr.events)


def test_cancel_not_confirmed_stops_worker(mgr, tmp_path, monkeypatch):
    monkeypatch.setattr(wmod, "CANCEL_GRACE_S", 2.0)
    rt = make_rt(tmp_path)
    pid = gen(mgr, rt).result["videos"][0]["worker_pid"]
    t0 = time.time()
    with pytest.raises(client.JobCancelled):
        gen(mgr, rt, prompt="MODE:hardhang", interrupted=lambda: time.time() - t0 > 1.0)
    assert _wait_gone(pid), "unresponsive worker was not stopped"
    assert gen(mgr, rt).result["worker"]["started_for_this_job"] is True


def test_timeout_cancels_job(mgr, tmp_path):
    rt = make_rt(tmp_path)
    with pytest.raises(client.JobTimeout):
        gen(mgr, rt, prompt="MODE:hang", timeout_s=2)
    assert gen(mgr, rt).result["worker"]["started_for_this_job"] is False  # clean cancel: still warm


def test_job_error_keeps_worker_fatal_error_replaces_it(mgr, tmp_path):
    rt = make_rt(tmp_path)
    pid = gen(mgr, rt).result["videos"][0]["worker_pid"]
    with pytest.raises(wmod.WorkerError, match="simulated failure"):
        gen(mgr, rt, prompt="MODE:fail")
    assert gen(mgr, rt).result["videos"][0]["worker_pid"] == pid
    with pytest.raises(wmod.WorkerError, match="CUDA"):
        gen(mgr, rt, prompt="MODE:cudafail")
    assert _wait_gone(pid)
    assert gen(mgr, rt).result["videos"][0]["worker_pid"] != pid


def test_worker_crash_fails_job_once_then_recovers(mgr, tmp_path):
    rt = make_rt(tmp_path)
    pid = gen(mgr, rt).result["videos"][0]["worker_pid"]
    with pytest.raises(wmod.WorkerError, match="exited during the job"):
        gen(mgr, rt, prompt="MODE:crash")
    assert _wait_gone(pid)
    nxt = gen(mgr, rt)
    assert nxt.result["worker"]["started_for_this_job"] is True  # the crashed job was not retried


def test_concurrent_jobs_are_serialised(mgr, tmp_path):
    rt = make_rt(tmp_path)
    gen(mgr, rt)
    results, errors = [], []

    def run(seed):
        try:
            results.append(gen(mgr, rt, seed=seed))
        except Exception as exc:  # pragma: no cover - reported below
            errors.append(exc)

    threads = [threading.Thread(target=run, args=(s,)) for s in (11, 12, 13)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(60)
    assert not errors and len(results) == 3
    assert sorted(o.result["worker"]["job_number_in_worker"] for o in results) == [2, 3, 4]
    assert len({o.result["videos"][0]["worker_pid"] for o in results}) == 1


def test_protocol_busy_duplicate_and_foreign_paths(mgr, tmp_path):
    rt = make_rt(tmp_path)
    gen(mgr, rt)
    w = mgr._worker
    job_id, job_dir = client.new_job_dir(rt)
    client._write_json(job_dir / "job.json", client.make_job(rt, job_id, "dense", [{"prompt": "MODE:hang", "seed": 1}]))
    rid = w.send("generate", job=client.host_to_runtime_path(rt, job_dir / "job.json"))

    def next_for(r, kinds, seconds=10):
        end = time.time() + seconds
        while time.time() < end:
            try:
                msg = w.messages.get(timeout=0.2)
            except queue.Empty:
                continue
            if msg.get("request_id") == r and msg.get("type") in kinds:
                return msg
        raise AssertionError(f"no {kinds} for {r}")

    next_for(rid, ("accepted",))
    busy = w.send("generate", job=client.host_to_runtime_path(rt, job_dir / "job.json"))
    assert next_for(busy, ("error",))["error_type"] == "Busy"
    w.send("status", request_id=busy)  # reused id
    assert next_for(busy, ("error",))["error_type"] == "DuplicateRequest"
    cancel = w.send("cancel", target=rid)
    next_for(cancel, ("accepted",))
    assert next_for(rid, ("cancelled",))
    outside = tmp_path / "elsewhere" / "job.json"
    outside.parent.mkdir()
    outside.write_text("{}", encoding="utf-8")
    bad = w.send("generate", job=str(outside))
    msg = next_for(bad, ("error",))
    assert "outside the jobs root" in msg["error"]


@pytest.mark.skipif(not POSIX, reason="checks process exit via /proc; on Windows hosts the worker runs inside WSL")
@pytest.mark.parametrize("busy", [False, True])
def test_worker_stops_when_comfyui_dies(tmp_path, fixture_mp4, busy):
    """A separate 'ComfyUI' process starts a worker, then dies without cleanup (os._exit)."""
    repo = Path(__file__).resolve().parents[1]
    script = tmp_path / "fake_comfy.py"
    script.write_text(
        textwrap.dedent(
            f"""
            import json, os, sys, threading, time
            sys.path.insert(0, {str(repo)!r}); sys.path.insert(0, {str(Path(__file__).resolve().parent)!r})
            from monarchrt_comfy import worker as wmod
            from test_worker import make_rt
            from pathlib import Path
            files = dict(wmod.CODE_FILES)
            files["fake_monarchrt_worker.py"] = Path({str(FAKE / "fake_monarchrt_worker.py")!r})
            files["fixture.mp4"] = Path({str(fixture_mp4)!r})
            wmod.CODE_FILES = files; wmod.WORKER_ENTRY = "fake_monarchrt_worker.py"
            m = wmod.WorkerManager()
            rt = make_rt(Path({str(tmp_path)!r}))
            out = m.generate(rt, "dense", [{{"prompt": "MODE:ok x", "seed": 1}}])
            print(json.dumps({{"pid": out.result["videos"][0]["worker_pid"]}}), flush=True)
            if {busy!r}:
                threading.Thread(target=lambda: m.generate(rt, "dense", [{{"prompt": "MODE:hang x", "seed": 2}}]), daemon=True).start()
                time.sleep(2)
            os._exit(0)  # ComfyUI crashes: no atexit, no unload
            """
        ),
        encoding="utf-8",
    )
    done = subprocess.run([sys.executable, str(script)], capture_output=True, text=True, timeout=120, env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
    pid = json.loads(done.stdout.strip().splitlines()[0])["pid"]
    assert _wait_gone(pid, 20), "worker survived its parent"


def test_exception_from_progress_callback_cancels_the_running_job(mgr, tmp_path):
    """ComfyUI's progress hook raises its interrupt exception inside on_progress; the worker must not keep the job."""
    rt = make_rt(tmp_path)
    pid = gen(mgr, rt).result["videos"][0]["worker_pid"]

    class HookInterrupt(Exception):
        pass

    def on_progress(done, total):
        if done >= 3:
            raise HookInterrupt()

    with pytest.raises(HookInterrupt):
        gen(mgr, rt, prompt="MODE:hang", on_progress=on_progress)
    assert any(e["event"] == "job_cancelled" and e["clean"] for e in mgr.events)
    nxt = gen(mgr, rt)
    assert nxt.result["videos"][0]["worker_pid"] == pid  # still warm, and not busy with the abandoned job
