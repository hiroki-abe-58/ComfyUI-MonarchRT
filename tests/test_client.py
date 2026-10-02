"""Pure checks of argv/env/job construction and result validation."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path, PureWindowsPath

import pytest

from monarchrt_comfy import client
from monarchrt_comfy.config import parse_runtime

REPO = Path(__file__).resolve().parents[1]
WSL = parse_runtime(
    "w",
    {
        "kind": "wsl",
        "distro": "Ubuntu-24.04",
        "python": "/opt/m/venv/bin/python",
        "upstream_dir": "/opt/m/MonarchRT",
        "models_dir": "/opt/m/models",
        "checkpoint": "/opt/m/ckpt.pt",
        "env": {"CC": "/opt/m/bin/cc", "TRITON_CACHE_DIR": "/opt/m/cache/triton"},
    },
)


def _runner():
    spec = importlib.util.spec_from_file_location("monarchrt_job_under_test", REPO / "runtime" / "monarchrt_job.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_wsl_path_translation():
    assert client.host_to_runtime_path(WSL, PureWindowsPath(r"E:\Comfy UI\temp\monarchrt\job-1")) == "/mnt/e/Comfy UI/temp/monarchrt/job-1"
    assert client.host_to_runtime_path(WSL, PureWindowsPath("C:/x/y.json")) == "/mnt/c/x/y.json"
    for bad in (r"\\server\share\x", "relative\\x", "/already/linux"):
        with pytest.raises(client.RuntimeJobError):
            client.host_to_runtime_path(WSL, PureWindowsPath(bad) if "\\" in bad else Path(bad))


def test_build_argv_is_a_fixed_list_without_shell(monkeypatch):
    monkeypatch.setitem(client.RUNTIME_SCRIPTS, client.JOB_SCRIPT, PureWindowsPath(r"E:\nodes\ComfyUI-MonarchRT\runtime\monarchrt_job.py"))
    monkeypatch.setenv("SystemRoot", r"C:\Windows")
    argv = client.build_argv(WSL, client.JOB_SCRIPT, "--watch-stdin", "/mnt/e/jobs/j/job.json")
    assert argv == [
        r"C:\Windows\System32\wsl.exe",
        "-d",
        "Ubuntu-24.04",
        "--cd",
        "/",
        "--exec",
        "/opt/m/venv/bin/python",
        "/mnt/e/nodes/ComfyUI-MonarchRT/runtime/monarchrt_job.py",
        "--watch-stdin",
        "/mnt/e/jobs/j/job.json",
    ]
    assert "--exec" in argv and not any(a in ("sh", "bash", "-c", "/bin/sh") for a in argv)
    with pytest.raises(KeyError):
        client.build_argv(WSL, "other_script.py")


def test_child_env_drops_secrets(monkeypatch):
    for k in ("GITHUB_TOKEN", "HF_TOKEN", "REGISTRY_ACCESS_TOKEN", "WANDB_API_KEY", "AWS_SECRET_ACCESS_KEY", "OPENAI_API_KEY"):
        monkeypatch.setenv(k, "secret-value")
    monkeypatch.setenv("WSLENV", "GITHUB_TOKEN/u")
    for rt in (WSL, parse_runtime("p", _posix_raw())):
        env = client.child_env(rt)
        assert "secret-value" not in env.values()
        assert not any("TOKEN" in k or "KEY" in k for k in env)
    assert client.child_env(WSL)["WSLENV"] == ""


def _posix_raw():
    return {"kind": "posix", "python": "/usr/bin/python3", "upstream_dir": "/u", "models_dir": "/m", "checkpoint": "/c.pt"}


def test_make_job_is_accepted_by_the_runner(tmp_path):
    runner = _runner()
    job = client.make_job(WSL, "job-20260101-000000-abcdef012345", "monarch_h2", [{"prompt": "a cat 猫", "seed": 7}])
    p = tmp_path / "job.json"
    p.write_text(json.dumps(job), encoding="utf-8")
    assert runner.load_job(p)["profile"] == "monarch_h2"


@pytest.mark.parametrize(
    "profile, videos",
    [
        ("monarch_h3", [{"prompt": "x", "seed": 1}]),
        ("dense", []),
        ("dense", [{"prompt": "x", "seed": 1}] * 9),
        ("dense", [{"prompt": " ", "seed": 1}]),
        ("dense", [{"prompt": "a\0b", "seed": 1}]),
        ("dense", [{"prompt": "x", "seed": -1}]),
        ("dense", [{"prompt": "x", "seed": True}]),
        ("dense", [{"prompt": "x" * 2001, "seed": 1}]),
    ],
)
def test_make_job_rejects(profile, videos):
    with pytest.raises(ValueError):
        client.make_job(WSL, "job-20260101-000000-abcdef012345", profile, videos)


@pytest.mark.parametrize(
    "patch",
    [
        {"job_id": "../../x"},
        {"profile": "dense; rm"},
        {"fps": 30},
        {"env": {"LD_PRELOAD": "/x.so"}},
        {"extra": 1},
        {"videos": [{"prompt": "x", "seed": 1, "cmd": "id"}]},
        {"offload_text_encoder": "true"},
    ],
)
def test_runner_rejects_tampered_jobs(tmp_path, patch):
    runner = _runner()
    job = client.make_job(WSL, "job-20260101-000000-abcdef012345", "dense", [{"prompt": "x", "seed": 1}])
    job.update(patch)
    p = tmp_path / "job.json"
    p.write_text(json.dumps(job), encoding="utf-8")
    with pytest.raises(runner.JobError):
        runner.load_job(p)


def test_runner_dispatch_check():
    runner = _runner()
    ok_m = {
        "monarch_kv_calls": 1050,
        "monarch_triton_calls": 1050,
        "monarch_torch_slow_calls": 0,
        "monarch_no_cache_calls": 0,
        "dense_self_attn_calls": 0,
        "flex_attn_calls": 0,
        "cross_attn_calls": 1050,
    }
    fwd = {"denoise": 28, "context": 7}
    runner._check_dispatch("monarch_h2", ok_m, fwd, 30)
    with pytest.raises(RuntimeError, match="dispatch"):
        runner._check_dispatch("monarch_h2", {**ok_m, "dense_self_attn_calls": 1}, fwd, 30)
    with pytest.raises(RuntimeError, match="dispatch"):
        runner._check_dispatch("monarch_h2", {**ok_m, "monarch_triton_calls": 0, "monarch_torch_slow_calls": 1050}, fwd, 30)
    with pytest.raises(RuntimeError, match="forward count"):
        runner._check_dispatch("monarch_h2", ok_m, {"denoise": 28, "context": 6}, 30)
    ok_d = {**ok_m, "monarch_kv_calls": 0, "monarch_triton_calls": 0, "dense_self_attn_calls": 1050}
    runner._check_dispatch("dense", ok_d, fwd, 30)
    with pytest.raises(RuntimeError):
        runner._check_dispatch("dense", ok_m, fwd, 30)


def test_new_job_dirs_are_unique_and_confined(fake_runtime):
    ids = set()
    for _ in range(5):
        job_id, d = client.new_job_dir(fake_runtime)
        assert d.parent == client.jobs_root(fake_runtime) and d.is_dir()
        ids.add(job_id)
    assert len(ids) == 5


def _fake_result(job_dir: Path, job: dict, rel="videos/00.mp4", frames=81):
    (job_dir / "videos").mkdir()
    mp4 = job_dir / "videos" / "00.mp4"
    mp4.write_bytes(b"not really an mp4")
    rec = {"file": rel, "sha256": hashlib.sha256(mp4.read_bytes()).hexdigest(), "rgb_frames": frames, "width": 832, "height": 480}
    (job_dir / "result.json").write_text(json.dumps({"job_id": job["job_id"], "profile": job["profile"], "videos": [rec]}), encoding="utf-8")
    return mp4


def test_load_result_validation(fake_runtime):
    job_id, d = client.new_job_dir(fake_runtime)
    job = client.make_job(fake_runtime, job_id, "dense", [{"prompt": "x", "seed": 1}])
    mp4 = _fake_result(d, job)
    assert client.load_result(d, job).videos == [mp4.resolve()]
    mp4.write_bytes(b"tampered")
    with pytest.raises(client.RuntimeJobError, match="sha256"):
        client.load_result(d, job)
    for kw, msg in (({"rel": "../job.json"}, "unexpected video path"), ({"frames": 80}, "geometry")):
        job_id, d = client.new_job_dir(fake_runtime)
        job = client.make_job(fake_runtime, job_id, "dense", [{"prompt": "x", "seed": 1}])
        _fake_result(d, job, **kw)
        with pytest.raises(client.RuntimeJobError, match=msg):
            client.load_result(d, job)
    with pytest.raises(client.RuntimeJobError, match="belong"):
        client.load_result(d, {**job, "job_id": "job-other-000000000000"})
