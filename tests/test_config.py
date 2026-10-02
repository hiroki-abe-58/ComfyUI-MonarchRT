from __future__ import annotations

import json
from pathlib import Path

import pytest

from monarchrt_comfy.config import ConfigError, load_runtimes, parse_runtime

REPO = Path(__file__).resolve().parents[1]
BASE = {
    "kind": "wsl",
    "distro": "Ubuntu-24.04",
    "python": "/opt/m/venv/bin/python",
    "upstream_dir": "/opt/m/MonarchRT",
    "models_dir": "/opt/m/models",
    "checkpoint": "/opt/m/models/checkpoints/self_forcing_dmd.pt",
}


def test_example_config_parses():
    data = json.loads((REPO / "examples" / "monarchrt.runtimes.example.json").read_text(encoding="utf-8"))
    rts = {k: parse_runtime(k, v) for k, v in data["runtimes"].items()}
    assert rts["wsl-reference"].kind == "wsl" and rts["linux-local"].kind == "posix"


@pytest.mark.parametrize(
    "patch, message",
    [
        ({"distro": "Ubuntu; rm -rf /"}, "distro"),
        ({"distro": "-d"}, "distro"),
        ({"python": "venv/bin/python"}, "python"),
        ({"python": "/opt/../bin/sh"}, "'..'"),
        ({"upstream_dir": "C:\\x"}, "upstream_dir"),
        ({"checkpoint": "/a\nb"}, "checkpoint"),
        ({"env": {"GITHUB_TOKEN": "x"}}, "env"),
        ({"env": {"LD_PRELOAD": "/x.so"}}, "env"),
        ({"env": {"CC": 1}}, "env"),
        ({"timeout_minutes": 0}, "timeout"),
        ({"timeout_minutes": True}, "timeout"),
        ({"offload_text_encoder": "yes"}, "offload"),
        ({"command": "/bin/sh -c x"}, "unknown keys"),
        ({"kind": "shell"}, "kind"),
        ({"jobs_dir": "relative/dir"}, "jobs_dir"),
    ],
)
def test_rejects_bad_runtime(patch, message):
    with pytest.raises(ConfigError, match=message):
        parse_runtime("rt", {**BASE, **patch})


def test_rejects_bad_ids_and_posix_distro():
    with pytest.raises(ConfigError, match="runtime id"):
        parse_runtime("../x", BASE)
    with pytest.raises(ConfigError, match="distro"):
        parse_runtime("rt", {**BASE, "kind": "posix"})


def test_load_runtimes_missing_file_is_empty(tmp_path):
    assert load_runtimes(tmp_path / "nope.json") == {}


def test_load_runtimes_broken_file_raises(tmp_path):
    p = tmp_path / "c.json"
    p.write_text("{not json", encoding="utf-8")
    with pytest.raises(ConfigError, match="JSON"):
        load_runtimes(p)
    p.write_text(json.dumps({"schema_version": 2, "runtimes": {}}), encoding="utf-8")
    with pytest.raises(ConfigError, match="schema_version"):
        load_runtimes(p)
    p.write_text(" " * (300 * 1024), encoding="utf-8")
    with pytest.raises(ConfigError, match="larger"):
        load_runtimes(p)


def test_env_override_path(runtimes_file):
    runtimes_file({"a": BASE})
    assert list(load_runtimes()) == ["a"]
