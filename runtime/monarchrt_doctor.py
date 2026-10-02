"""Check a MonarchRT runtime environment (runs inside the runtime, like monarchrt_job.py).

Usage: monarchrt_doctor.py <request.json>

The request has the same paths/env as a job plus two switches:
  {"schema_version": 1, "job_id": ..., "upstream_dir": ..., "models_dir": ..., "checkpoint": ...,
   "env": {...}, "verify_sha256": false, "kernel_check": false}

Writes <request dir>/doctor.json and prints it. Exit 0 when no check failed.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
from monarchrt_job import ENV_KEYS, SECRET_RE, _proc_stat  # noqa: E402

UPSTREAM_COMMIT = "34867c041ff7d70c699a9149f84857058d8ada92"
# sha256 of the upstream files this integration depends on, at UPSTREAM_COMMIT, with line endings
# normalised to LF (a Windows git with core.autocrlf=true checks the same commit out with CRLF)
UPSTREAM_FILES = {
    "wan/modules/monarch_attn.py": "6c67b0b90d3a5df2c89e5ea9ea7dbaa1a8770fddbefa000d1823f14dd607f901",
    "wan/modules/causal_model.py": "250df5455fae999eb6e336f741c1194603735ebb33cedbdb045afa7c84bfb304",
    "wan/modules/model.py": "7a12954d6cf35bfceab873540d19c2318dae897c53efd20d885ac188e4a6457c",
    "wan/modules/attention.py": "4532ee4b94de03a9cd3d8ef18999d75d00353c8ba114fc58754e873569fe0fa8",
    "wan/modules/t5.py": "8b0cebf3192c542f92a344255a06c203df3ba24160715899a055cc8de0cd930f",
    "wan/modules/vae.py": "2f58c33ed9b4640ca5191cd0ccbc744d9f0bdb72f4334983f38dcfc5c4ffdd31",
    "pipeline/causal_inference.py": "9ead04b9315b75096cc7b70ee311c3de333bd33c2018e9be2db783aa35784da1",
    "utils/wan_wrapper.py": "bf376ea942e53597bd3fe710fa07eef057d98105e8d177e92fa842e82ec2696c",
    "configs/default_config.yaml": "8ec1cc2c5169fa8998adf9503f9ff3f1f75edafd145ca30930c9757a71a1ef5e",
    "configs/self_forcing_dmd.yaml": "55b13d0eccda6b51ee4838a2b2ab4ca1a0a00e2198d21d0ca5610ef421749bc5",
    "configs/self_forcing_monarch_dmd.yaml": "7f2149afeb15034a9b4e24cf7bd63ce21988c2acd06e0091c15a8626afc4d8a5",
}
# Wan-AI/Wan2.1-T2V-1.3B @ 37ec512624d61f7aa208f7ea8140a131f93afc9a (under models_dir)
MODEL_FILES = {
    "wan_models/Wan2.1-T2V-1.3B/config.json": (249, "ab37994c43740513f94b3ba6233a784035a67b43c8cde83c8f31aa90468c67ce"),
    "wan_models/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors": (5676070424, "96b6b242ca1c2f24e9d02cd6596066fab6d310e2d7538f33ae267cb18d957e8f"),
    "wan_models/Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth": (11361920418, "7cace0da2b446bbbbc57d031ab6cf163a3d59b366da94e5afe36745b746fd81d"),
    "wan_models/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth": (507609880, "38071ab59bd94681c686fa51d75a1968f64e470262043be31f7a094e442fd981"),
    "wan_models/Wan2.1-T2V-1.3B/google/umt5-xxl/spiece.model": (4548313, "e3909a67b780650b35cf529ac782ad2b6b26e6d1f849d3fbb6a872905f452458"),
    "wan_models/Wan2.1-T2V-1.3B/google/umt5-xxl/tokenizer.json": (16837417, "6e197b4d3dbd71da14b4eb255f4fa91c9c1f2068b20a2de2472967ca3d22602b"),
    "wan_models/Wan2.1-T2V-1.3B/google/umt5-xxl/tokenizer_config.json": (61728, "ed9a3a8b0faa71a70a32847e0435fe036e6e112d4df4edb7bb48a921e344dc05"),
    "wan_models/Wan2.1-T2V-1.3B/google/umt5-xxl/special_tokens_map.json": (6623, "7b8a9f5040adb67b5805abdfd42c1f8d0f3d0e711f10726580eb3789cd0ad61d"),
}
# gdhe17/Self-Forcing @ 2f8b779212da279d212c22a509b66ad6552f350e
CHECKPOINT = (5676252553, "a0413986d9734e02c09504e1520f5697ba6df731bb2f0f35577485e9cc8f56a3")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha256_text(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def _git_head(repo: Path) -> str | None:
    git = repo / ".git"
    try:
        head = (git / "HEAD").read_text(encoding="utf-8").strip()
        if not head.startswith("ref: "):
            return head
        ref = head[5:]
        if (git / ref).is_file():
            return (git / ref).read_text(encoding="utf-8").strip()
        for line in (git / "packed-refs").read_text(encoding="utf-8").splitlines():
            if line.endswith(" " + ref):
                return line.split()[0]
    except OSError:
        return None
    return None


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: monarchrt_doctor.py <request.json>", file=sys.stderr)
        return 2
    req_path = Path(argv[1]).resolve()
    if hasattr(os, "setsid"):
        try:
            os.setsid()
        except OSError:
            pass
    me = _proc_stat(os.getpid())
    record = {"pid": os.getpid(), "pgid": os.getpgid(0) if hasattr(os, "getpgid") else None, "starttime": int(me[19]) if me else None}
    (req_path.parent / "runner.pid").write_text(json.dumps(record), encoding="utf-8")
    req = json.loads(req_path.read_text(encoding="utf-8"))
    if not isinstance(req, dict) or req.get("schema_version") != 1:
        print(json.dumps({"error": "unsupported request"}))
        return 2
    env = req.get("env", {})
    if not isinstance(env, dict) or set(env) - ENV_KEYS:
        print(json.dumps({"error": f"env may only contain {sorted(ENV_KEYS)}"}))
        return 2
    for key in list(os.environ):
        if SECRET_RE.search(key):
            del os.environ[key]
    os.environ.update({k: str(v) for k, v in env.items()})
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    checks: list[dict] = []

    def add(name: str, status: str, detail) -> None:
        checks.append({"name": name, "status": status, "detail": detail})

    report: dict = {"schema_version": 1, "platform": platform.platform(), "python": sys.version.split()[0], "checks": checks}

    # --- upstream checkout ---
    upstream = Path(str(req.get("upstream_dir", ""))).resolve()
    head = _git_head(upstream)
    add("upstream_commit", "ok" if head == UPSTREAM_COMMIT else "warn", {"expected": UPSTREAM_COMMIT, "found": head})
    bad = {}
    for rel, digest in UPSTREAM_FILES.items():
        p = upstream / rel
        got = _sha256_text(p) if p.is_file() else None
        if got != digest:
            bad[rel] = "missing" if got is None else "modified"
    add("upstream_files", "fail" if bad else "ok", bad or f"{len(UPSTREAM_FILES)} files match the pinned commit")

    # --- model files ---
    verify = bool(req.get("verify_sha256", False))
    models = Path(str(req.get("models_dir", ""))).resolve()
    ckpt = Path(str(req.get("checkpoint", ""))).resolve()
    problems = {}
    for rel, (size, digest) in list(MODEL_FILES.items()) + [("<checkpoint>", CHECKPOINT)]:
        p = ckpt if rel == "<checkpoint>" else models / rel
        if not p.is_file():
            problems[rel] = "missing"
        elif p.stat().st_size != size:
            problems[rel] = f"size {p.stat().st_size} != {size}"
        elif verify and _sha256(p) != digest:
            problems[rel] = "sha256 mismatch"
    add("model_files", "fail" if problems else "ok", problems or ("sizes and sha256 match" if verify else "sizes match (sha256 not checked)"))

    # --- python packages and GPU ---
    try:
        import torch

        report["torch"] = torch.__version__
        report["torch_cuda"] = torch.version.cuda
        if not torch.cuda.is_available():
            add("cuda", "fail", "torch.cuda.is_available() is False")
        else:
            cap = torch.cuda.get_device_capability(0)
            free, total = torch.cuda.mem_get_info()
            add(
                "cuda",
                "ok",
                {
                    "device": torch.cuda.get_device_name(0),
                    "capability": f"{cap[0]}.{cap[1]}",
                    "free_gb": round(free / 2**30, 1),
                    "total_gb": round(total / 2**30, 1),
                },
            )
            if total < 40 * 2**30 and not req.get("offload_text_encoder", False):
                # upstream inference.py offloads the 11 GB UMT5-XXL encoder below 40 GB free VRAM; without it a
                # 32 GB card runs at the limit and Windows/WSL may spill into shared system memory (slow)
                add("vram", "warn", "less than 40 GB VRAM: set offload_text_encoder: true in the runtime config")
    except Exception as exc:
        add("torch", "fail", f"{type(exc).__name__}: {exc}")
    for mod in ("triton", "flash_attn", "flashinfer", "diffusers", "transformers", "omegaconf", "av", "einops"):
        try:
            m = __import__(mod)
            add(f"import:{mod}", "ok", getattr(m, "__version__", "?"))
        except Exception as exc:
            add(f"import:{mod}", "fail", f"{type(exc).__name__}: {str(exc)[:300]}")

    # --- a tiny Triton kernel (exercises the C compiler used for Triton's launcher) ---
    try:
        import torch
        import triton
        import triton.language as tl

        @triton.jit
        def _add(x, y, o, n, B: tl.constexpr):
            i = tl.program_id(0) * B + tl.arange(0, B)
            m = i < n
            tl.store(o + i, tl.load(x + i, mask=m) + tl.load(y + i, mask=m), mask=m)

        a = torch.randn(4096, device="cuda")
        b = torch.randn_like(a)
        out = torch.empty_like(a)
        s = time.time()
        _add[(4,)](a, b, out, 4096, B=1024)
        torch.cuda.synchronize()
        add("triton_kernel", "ok" if torch.allclose(out, a + b) else "fail", {"seconds": round(time.time() - s, 2), "CC": os.environ.get("CC", "")})
    except Exception as exc:
        add("triton_kernel", "fail", f"{type(exc).__name__}: {str(exc)[:500]}")

    # --- optional: Monarch Triton kernel vs the upstream torch reference at inference geometry ---
    if req.get("kernel_check", False) and not any(c["status"] == "fail" for c in checks if c["name"] in ("upstream_files", "cuda")):
        try:
            import torch

            sys.path.insert(0, str(upstream))
            from wan.modules.monarch_attn import monarch_attn_with_kv_cache, monarch_attn_with_kv_cache_ref

            torch.manual_seed(0)
            H, D, FH, FW, BLOCK, FR = 12, 128, 30, 52, 3, 6
            res = []
            for h_reduce in (2, 1):
                q = torch.randn(1, BLOCK * FH * FW, H, D, device="cuda", dtype=torch.bfloat16)
                nk, nv = torch.randn_like(q), torch.randn_like(q)
                ck = torch.randn(1, FR * FH * FW, H, D, device="cuda", dtype=torch.bfloat16)
                cv = torch.randn_like(ck)
                st, en = (FR - BLOCK) * FH * FW, FR * FH * FW
                with torch.no_grad():
                    t0 = time.time()
                    o = monarch_attn_with_kv_cache(q, ck.clone(), cv.clone(), nk, nv, st, en, 1, h_reduce, 1, FH, FW).float()
                    torch.cuda.synchronize()
                    dt = time.time() - t0
                    r = monarch_attn_with_kv_cache_ref(q, ck.clone(), cv.clone(), nk, nv, st, en, 1, h_reduce, 1, FH, FW, grad_only_new_kv=True).float()
                cos = torch.nn.functional.cosine_similarity(o.flatten(), r.flatten(), dim=0).item()
                ok = bool(torch.isfinite(o).all()) and torch.allclose(o, r, atol=2e-2, rtol=2e-2)
                res.append(
                    {"h_reduce": h_reduce, "cosine": round(cos, 6), "max_abs": (o - r).abs().max().item(), "allclose_bf16": ok, "first_call_s": round(dt, 1)}
                )
            add("monarch_kernel", "ok" if all(x["allclose_bf16"] for x in res) else "fail", res)
        except Exception as exc:
            add("monarch_kernel", "fail", f"{type(exc).__name__}: {str(exc)[:500]}")

    report["overall"] = "fail" if any(c["status"] == "fail" for c in checks) else ("warn" if any(c["status"] == "warn" for c in checks) else "ok")
    text = json.dumps(report, indent=1, ensure_ascii=False)
    tmp = req_path.parent / "doctor.json.tmp"
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(req_path.parent / "doctor.json")
    print(text)
    return 0 if report["overall"] != "fail" else 1


if __name__ == "__main__":
    if not re.match(r"^3\.(1[0-9])", platform.python_version()):
        print(json.dumps({"error": f"python {platform.python_version()} is not supported (3.10+)"}))
        sys.exit(2)
    sys.exit(main(sys.argv))
