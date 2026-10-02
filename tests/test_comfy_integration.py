"""Against a real ComfyUI checkout on CPU: registration, validation, execution with the fake runtime."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

FAKE_RUNTIME = Path(__file__).resolve().parent / "fake_runtime"

pytestmark = pytest.mark.comfy
NODE_IDS = ("MonarchRTRuntime", "MonarchRTGenerate", "MonarchRTDoctor")


def _mod(comfyui):
    return sys.modules[comfyui["nodes"].NODE_CLASS_MAPPINGS["MonarchRTGenerate"].__module__]


def _use_fake(mod, fake_runtime, monkeypatch):
    # ComfyUI imported its own copy of the package; point that copy at the fake runtime
    monkeypatch.setitem(mod.client.RUNTIME_SCRIPTS, mod.client.JOB_SCRIPT, FAKE_RUNTIME / "monarchrt_job.py")
    monkeypatch.setattr(mod, "_resolve", lambda rid: fake_runtime)


def test_nodes_registered_without_heavy_imports(comfyui):
    for nid in NODE_IDS:
        assert nid in comfyui["nodes"].NODE_CLASS_MAPPINGS
    # importing the node package must not pull in the runtime side
    for heavy in ("triton", "flash_attn", "flashinfer", "wan", "pipeline", "omegaconf"):
        assert heavy not in sys.modules, heavy


def test_no_runtime_configured(comfyui, tmp_path, monkeypatch):
    monkeypatch.setenv("COMFYUI_MONARCHRT_CONFIG", str(tmp_path / "missing.json"))
    mod = _mod(comfyui)
    assert mod.MonarchRTRuntime.INPUT_TYPES()["required"]["runtime_id"][0] == [mod.NO_RUNTIME]
    with pytest.raises(ValueError, match="not configured"):
        mod.MonarchRTRuntime().select(mod.NO_RUNTIME)
    out = mod.MonarchRTDoctor().check(mod.NO_RUNTIME, False, False)
    assert json.loads(out["result"][0])["overall"] == "fail"


def _api_prompt(runtime_id="fake", attention="monarch_h2"):
    return {
        "1": {"class_type": "MonarchRTRuntime", "inputs": {"runtime_id": runtime_id}},
        "2": {"class_type": "MonarchRTGenerate", "inputs": {"runtime": ["1", 0], "prompt": "MODE:ok a fox", "seed": 3, "attention": attention, "videos": 1}},
        "3": {"class_type": "SaveVideo", "inputs": {"video": ["2", 0], "filename_prefix": "monarchrt/test", "format": "auto", "codec": "auto"}},
    }


def test_validate_prompt(comfyui, runtimes_file):
    import execution

    runtimes_file({"fake": {"kind": "posix", "python": "/x/python", "upstream_dir": "/u", "models_dir": "/m", "checkpoint": "/c.pt"}})
    ok = __import__("asyncio").run(execution.validate_prompt("p1", _api_prompt(), None))
    assert ok[0] is True and not ok[3], ok
    bad = __import__("asyncio").run(execution.validate_prompt("p2", _api_prompt(attention="sparse_magic"), None))
    assert bad[0] is False or bad[3]
    bad = __import__("asyncio").run(execution.validate_prompt("p3", _api_prompt(runtime_id="not-registered"), None))
    assert bad[0] is False or bad[3]


def test_generate_returns_real_video(comfyui, fake_runtime, monkeypatch):
    mod = _mod(comfyui)
    _use_fake(mod, fake_runtime, monkeypatch)
    videos, report = mod.MonarchRTGenerate().generate({"runtime_id": "fake"}, "MODE:ok a fox", 5, "monarch_h2", 2)
    from comfy_api.latest import InputImpl

    assert len(videos) == 2 and all(isinstance(v, InputImpl.VideoFromFile) for v in videos)
    assert videos[0].get_dimensions() == (832, 480)
    comps = videos[1].get_components()
    assert comps.images.shape[0] == 81 and float(comps.frame_rate) == 16.0
    rep = json.loads(report)
    assert rep["profile"] == "monarch_h2" and rep["training_free"] is True and [v["seed"] for v in rep["videos"]] == [5, 6]


def test_generate_interrupt_maps_to_comfy_interrupt(comfyui, fake_runtime, monkeypatch):
    import comfy.model_management as mm

    mod = _mod(comfyui)
    _use_fake(mod, fake_runtime, monkeypatch)
    monkeypatch.setattr(mm, "processing_interrupted", lambda: True)
    with pytest.raises(mm.InterruptProcessingException):
        mod.MonarchRTGenerate().generate({"runtime_id": "fake"}, "MODE:hang", 1, "dense", 1)
    job_dir = max(mod.client.jobs_root(fake_runtime).iterdir(), key=lambda p: p.stat().st_mtime)
    assert (job_dir / "stop_report.json").exists()
    gc = job_dir / "grandchild.pid"
    if gc.exists() and sys.platform == "win32":
        import os

        try:
            os.kill(int(gc.read_text()), 9)
        except OSError:
            pass
