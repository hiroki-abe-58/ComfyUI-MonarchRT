"""Run one structured MonarchRT / Self-Forcing generation job.

This script runs inside the *runtime* environment (the one with PyTorch, Triton,
flashinfer and an upstream MonarchRT checkout), never inside ComfyUI. ComfyUI
writes a job JSON into a fresh job directory and starts:

    <runtime python> monarchrt_job.py <job.json>

The upstream code is imported unmodified. This file adds:

- schema/type/limit checks of the job,
- a guard that makes every ``torch.load`` use ``weights_only=True``,
- strict loading of the Self-Forcing EMA generator weights with evidence that
  they were applied,
- per-video seeding so dense and Monarch runs see identical noise,
- counters for generator forwards (denoise vs. KV-context update) and for the
  attention implementation actually dispatched (Monarch Triton, Monarch torch,
  dense self-attention, cross-attention),
- timings per phase and CUDA allocator peaks,
- MP4 + metadata output confined to the job directory.

Usage: monarchrt_job.py [--watch-stdin] <job.json>

Exit codes: 0 success, 2 invalid job, 3 runtime failure, 4 cancelled.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import threading
import time
from pathlib import Path

SCHEMA_VERSION = 1
PROFILES = {
    # profile id -> (upstream config file, monarch overrides or None for dense)
    "dense": ("self_forcing_dmd.yaml", None),
    "monarch_h2": ("self_forcing_monarch_dmd.yaml", {"enable": True, "num_iters": 1, "f_tied": 1, "h_reduce": 2, "w_reduce": 1}),
    "monarch_h1": ("self_forcing_monarch_dmd.yaml", {"enable": True, "num_iters": 1, "f_tied": 1, "h_reduce": 1, "w_reduce": 1}),
}
LATENT_FRAMES = 21  # 21 latent frames -> 1 + 4 * 20 = 81 RGB frames (Wan VAE temporal stride 4)
LATENT_SHAPE = (16, 60, 104)  # channels, 480/8, 832/8
# 21 latent frames in blocks of 3 -> 7 blocks x (4 denoising steps + 1 clean-context KV update)
FORWARDS_PER_VIDEO = 35
MAX_VIDEOS = 8
MAX_PROMPT_CHARS = 2000
ENV_KEYS = {"CC", "TRITON_CACHE_DIR", "TORCHINDUCTOR_CACHE_DIR", "XDG_CACHE_HOME", "HF_HOME", "FLASHINFER_WORKSPACE_BASE", "CUDA_VISIBLE_DEVICES"}
SECRET_RE = re.compile(r"(TOKEN|SECRET|PASSWORD|API_KEY|ACCESS_KEY|CREDENTIAL|WANDB)", re.I)


class JobError(ValueError):
    pass


_EVENTS_PATH: Path | None = None


def _log(event: str, **fields) -> None:
    """One JSON object per line on stdout and, once the job directory is known, in events.jsonl."""
    line = json.dumps({"event": event, "t": round(time.time(), 3), **fields}, ensure_ascii=False)
    print(line, flush=True)
    if _EVENTS_PATH is not None:
        with open(_EVENTS_PATH, "a", encoding="utf-8") as f:
            f.write(line + "\n")


def load_job(path: Path) -> dict:
    """Parse and validate the job file. Raises JobError."""
    if path.stat().st_size > 256 * 1024:
        raise JobError("job file too large")
    try:
        job = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise JobError(f"job is not valid UTF-8 JSON: {exc}") from exc
    if not isinstance(job, dict) or job.get("schema_version") != SCHEMA_VERSION:
        raise JobError("unsupported job schema")
    allowed = {"schema_version", "job_id", "profile", "videos", "upstream_dir", "models_dir", "checkpoint", "fps", "env", "offload_text_encoder"}
    extra = set(job) - allowed
    if extra:
        raise JobError(f"unknown job keys: {sorted(extra)}")
    if not isinstance(job.get("job_id"), str) or not re.fullmatch(r"[A-Za-z0-9_-]{8,80}", job["job_id"]):
        raise JobError("invalid job_id")
    if job.get("profile") not in PROFILES:
        raise JobError(f"profile must be one of {sorted(PROFILES)}")
    videos = job.get("videos")
    if not isinstance(videos, list) or not 1 <= len(videos) <= MAX_VIDEOS:
        raise JobError(f"videos must be a list of 1..{MAX_VIDEOS} items")
    for i, v in enumerate(videos):
        if not isinstance(v, dict) or set(v) - {"prompt", "seed"}:
            raise JobError(f"video {i}: expected {{prompt, seed}}")
        p, s = v.get("prompt"), v.get("seed")
        if not isinstance(p, str) or not p.strip() or len(p) > MAX_PROMPT_CHARS or "\x00" in p:
            raise JobError(f"video {i}: prompt must be 1..{MAX_PROMPT_CHARS} characters")
        if not isinstance(s, int) or isinstance(s, bool) or not 0 <= s <= 2**31 - 1:
            raise JobError(f"video {i}: seed must be an integer in [0, 2^31-1]")
    for key in ("upstream_dir", "models_dir", "checkpoint"):
        if not isinstance(job.get(key), str) or not job[key]:
            raise JobError(f"{key} must be a non-empty string")
    if job.get("fps", 16) not in (16,):
        raise JobError("fps must be 16 (the upstream output rate)")
    env = job.get("env", {})
    if not isinstance(env, dict) or set(env) - ENV_KEYS or not all(isinstance(v, str) for v in env.values()):
        raise JobError(f"env may only contain {sorted(ENV_KEYS)} as strings")
    if not isinstance(job.get("offload_text_encoder", False), bool):
        raise JobError("offload_text_encoder must be a boolean")
    return job


def _install_cancel_watch(job_dir: Path, watch_stdin: bool) -> None:
    """Exit promptly if a CANCEL file appears or (with --watch-stdin) the parent closes our stdin."""

    def stdin_eof():
        try:
            while os.read(0, 4096):  # raw fd: no buffered-reader lock to trip interpreter shutdown
                pass
        except OSError:
            pass
        if not (job_dir / "result.json").exists():
            _log("cancelled", reason="parent closed stdin")
            _stop_own_group()
            os._exit(4)

    def cancel_file():
        while True:
            if (job_dir / "CANCEL").exists():
                _log("cancelled", reason="cancel file")
                _stop_own_group()
                os._exit(4)
            time.sleep(0.5)

    if watch_stdin:
        threading.Thread(target=stdin_eof, daemon=True).start()
    threading.Thread(target=cancel_file, daemon=True).start()


def _proc_stat(pid: int) -> list[str] | None:
    """Fields of /proc/<pid>/stat after the command name (index 0 = state, 2 = pgrp, 3 = session, 19 = starttime)."""
    try:
        with open(f"/proc/{pid}/stat", "rb") as f:
            stat = f.read().decode("utf-8", "replace")
    except OSError:
        return None
    return stat[stat.rfind(")") + 2 :].split()


def _stop_own_group() -> None:
    """SIGKILL every other member of this runner's process group (compile workers, compiler processes...).

    Only when the runner leads its own session, i.e. setsid() succeeded or the parent started it in a new session.
    """
    me = os.getpid()
    if not hasattr(os, "getsid") or os.getpgid(0) != me or os.getsid(0) != me or not os.path.isdir("/proc"):
        return  # not our own session (e.g. started from an interactive shell pipeline): leave the group alone
    for entry in os.listdir("/proc"):
        if entry.isdigit() and int(entry) != me:
            st = _proc_stat(int(entry))
            if st and int(st[2]) == me:
                try:
                    os.kill(int(entry), 9)
                except OSError:
                    pass


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _check_dispatch(profile: str, counts: dict, fwd: dict, layers: int) -> None:
    """Fail loudly unless every self-attention call took the path the profile asks for (no silent fallback)."""
    if fwd != {"denoise": 28, "context": 7}:
        raise RuntimeError(f"unexpected generator forward count {fwd} (expected 28 denoise + 7 context)")
    calls = FORWARDS_PER_VIDEO * layers
    if profile == "dense":
        want = {"dense_self_attn_calls": calls, "monarch_kv_calls": 0, "monarch_triton_calls": 0, "monarch_torch_slow_calls": 0}
    else:
        want = {"monarch_kv_calls": calls, "monarch_triton_calls": calls, "monarch_torch_slow_calls": 0, "dense_self_attn_calls": 0}
    want.update(monarch_no_cache_calls=0, flex_attn_calls=0, cross_attn_calls=calls)
    bad = {k: (counts.get(k), v) for k, v in want.items() if counts.get(k) != v}
    if bad:
        raise RuntimeError(f"attention dispatch does not match profile {profile!r}: {bad} (got, expected)")


def _peak_rss_bytes() -> int | None:
    try:
        import resource

        return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024  # Linux reports KiB
    except Exception:
        return None


def main(argv: list[str]) -> int:
    global _EVENTS_PATH
    sys.dont_write_bytecode = True  # never leave __pycache__ in the upstream checkout
    args = argv[1:]
    watch_stdin = "--watch-stdin" in args
    args = [a for a in args if a != "--watch-stdin"]
    if len(args) != 1:
        print("usage: monarchrt_job.py [--watch-stdin] <job.json>", file=sys.stderr)
        return 2
    job_path = Path(args[0]).resolve()
    job_dir = job_path.parent
    _EVENTS_PATH = job_dir / "events.jsonl"
    try:
        job = load_job(job_path)
    except (JobError, OSError) as exc:
        _log("invalid_job", error=str(exc))
        return 2

    # Environment: drop anything that looks like a credential, then apply the allowlisted runtime settings.
    for key in list(os.environ):
        if SECRET_RE.search(key):
            del os.environ[key]
    os.environ.update(job.get("env", {}))
    os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    # Everything is loaded from local files; never reach out to the Hugging Face Hub.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    if hasattr(os, "setsid"):
        try:
            os.setsid()  # own process group so the parent can stop the whole tree
        except OSError:
            pass
    me = _proc_stat(os.getpid())
    record = {"pid": os.getpid(), "pgid": os.getpgid(0) if hasattr(os, "getpgid") else None, "starttime": int(me[19]) if me else None}
    (job_dir / "runner.pid").write_text(json.dumps(record), encoding="utf-8")
    _install_cancel_watch(job_dir, watch_stdin)
    _log("start", job_id=job["job_id"], profile=job["profile"], videos=len(job["videos"]))

    try:
        return _run(job, job_dir)
    except Exception as exc:  # report and fail; never write a result on failure
        import traceback

        _log("failed", error_type=type(exc).__name__, error=str(exc)[:2000], traceback=traceback.format_exc()[-4000:])
        return 3


def _run(job: dict, job_dir: Path) -> int:
    t_proc = time.time()
    upstream = Path(job["upstream_dir"]).resolve()
    models = Path(job["models_dir"]).resolve()
    ckpt = Path(job["checkpoint"]).resolve()
    for p, what in (
        (upstream / "pipeline" / "causal_inference.py", "upstream checkout"),
        (models / "wan_models" / "Wan2.1-T2V-1.3B" / "config.json", "Wan2.1-T2V-1.3B"),
        (ckpt, "checkpoint"),
    ):
        if not p.is_file():
            raise FileNotFoundError(f"{what} not found")
    sys.path.insert(0, str(upstream))
    os.chdir(models)  # upstream resolves wan_models/... relative to the working directory

    import torch

    _orig_load = torch.load

    def _safe_load(*args, **kwargs):
        kwargs["weights_only"] = True  # upstream passes weights_only=False for T5; refuse pickled code objects
        kwargs.setdefault("mmap", True)  # page-cache backed instead of a second full copy in RAM
        return _orig_load(*args, **kwargs)

    torch.load = _safe_load
    import av
    import triton
    import utils.wan_wrapper as wan_wrapper
    import wan.modules.causal_model as causal_model
    import wan.modules.model as wan_model
    import wan.modules.monarch_attn as monarch_attn
    from omegaconf import OmegaConf
    from pipeline import CausalInferencePipeline

    # Upstream builds the UMT5-XXL encoder in fp32 on the CPU (~23 GB) and then loads the bf16 file into it;
    # the whole pipeline is cast to bf16 right after. Build it on the meta device and adopt the bf16 tensors
    # instead: the resulting weights are bit-identical, without the 23 GB fp32 intermediate.
    _orig_umt5 = wan_wrapper.umt5_xxl

    def _umt5_meta(**kwargs):
        kwargs.update(device="meta", dtype=torch.bfloat16)
        model = _orig_umt5(**kwargs)
        model.load_state_dict = lambda sd, strict=True: torch.nn.Module.load_state_dict(model, sd, strict=strict, assign=True)
        return model

    wan_wrapper.umt5_xxl = _umt5_meta

    versions = {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "triton": triton.__version__,
        "device": torch.cuda.get_device_name(0),
        "capability": ".".join(map(str, torch.cuda.get_device_capability(0))),
    }
    try:
        import flashinfer

        versions["flashinfer"] = getattr(flashinfer, "__version__", "unknown")
    except Exception as exc:
        raise RuntimeError(f"flashinfer import failed: {exc}") from exc

    # --- dispatch counters (wrap the names the upstream modules call) ---
    import wan.modules.attention as wan_attention

    counts = {
        "monarch_kv_calls": 0,  # causal self-attention with KV cache, Monarch branch
        "monarch_triton_calls": 0,  # ... of which reached the fused Triton kernel (num_iters == 1)
        "monarch_torch_slow_calls": 0,  # ... of which used the pure-torch iterative path (num_iters > 1)
        "monarch_no_cache_calls": 0,  # Monarch without KV cache (training path; expected 0)
        "dense_self_attn_calls": 0,  # causal self-attention with KV cache, dense branch
        "flex_attn_calls": 0,  # dense without KV cache (training path; expected 0)
        "cross_attn_calls": 0,  # text cross-attention (always dense flash-attn)
    }

    def _wrap(fn, key):
        def inner(*a, **k):
            counts[key] += 1
            return fn(*a, **k)

        return inner

    # causal_model / model bind these names at import time and look them up as module globals on each call
    causal_model.monarch_attn_with_kv_cache = _wrap(causal_model.monarch_attn_with_kv_cache, "monarch_kv_calls")
    monarch_attn._attention_with_cache.apply = _wrap(monarch_attn._attention_with_cache.apply, "monarch_triton_calls")
    monarch_attn.monarch_attn_slow = _wrap(monarch_attn.monarch_attn_slow, "monarch_torch_slow_calls")
    causal_model.monarch_attn = _wrap(causal_model.monarch_attn, "monarch_no_cache_calls")
    causal_model.attention = _wrap(causal_model.attention, "dense_self_attn_calls")
    causal_model.flex_attention = _wrap(causal_model.flex_attention, "flex_attn_calls")
    wan_model.flash_attention = _wrap(wan_model.flash_attention, "cross_attn_calls")
    attn_backend = {
        "flash_attn_2": bool(wan_attention.FLASH_ATTN_2_AVAILABLE),
        "flash_attn_3": bool(wan_attention.FLASH_ATTN_3_AVAILABLE),
    }
    if not attn_backend["flash_attn_2"] and not attn_backend["flash_attn_3"]:
        raise RuntimeError("flash-attn is required by the upstream cross-attention (wan/modules/attention.py)")
    import flash_attn

    versions["flash_attn"] = flash_attn.__version__

    # --- config ---
    cfg_file, overrides = PROFILES[job["profile"]]
    config = OmegaConf.merge(OmegaConf.load(upstream / "configs" / "default_config.yaml"), OmegaConf.load(upstream / "configs" / cfg_file))
    if overrides is not None:
        for k, v in overrides.items():
            config.monarch_args[k] = v
    margs = OmegaConf.to_container(config.get("monarch_args", {}), resolve=True) or {}
    expected = overrides or {"enable": False}
    for k, v in expected.items():
        if margs.get(k) != v:
            raise RuntimeError(f"monarch_args.{k} = {margs.get(k)!r}, expected {v!r}")
    effective = {
        "config_file": cfg_file,
        "monarch_args": margs,
        "denoising_step_list": list(config.denoising_step_list),
        "warp_denoising_step": bool(config.warp_denoising_step),
        "num_frame_per_block": int(config.num_frame_per_block),
        "context_noise": int(config.context_noise),
        "timestep_shift": float(config.model_kwargs.timestep_shift),
        "offload_text_encoder": bool(job.get("offload_text_encoder", False)),
    }

    # --- model load ---
    torch.set_grad_enabled(False)
    device = torch.device("cuda")
    t0 = time.time()
    pipeline = CausalInferencePipeline(config, device=device)
    meta_left = [n for n, p in list(pipeline.named_parameters()) + list(pipeline.named_buffers()) if p.is_meta]
    if meta_left:
        raise RuntimeError(f"{len(meta_left)} tensors were never loaded, e.g. {meta_left[:3]}")
    state = torch.load(str(ckpt), map_location="cpu")
    if "generator_ema" not in state:
        raise RuntimeError(f"checkpoint has no 'generator_ema' (keys: {sorted(state)[:6]})")
    ema = state["generator_ema"]
    gen_sd = pipeline.generator.state_dict()
    shape_bad = [k for k, v in ema.items() if k in gen_sd and tuple(gen_sd[k].shape) != tuple(v.shape)]
    if shape_bad:
        raise RuntimeError(f"{len(shape_bad)} EMA tensors have mismatched shapes, e.g. {shape_bad[:3]}")
    probe_key = next(k for k in ema if k.endswith("blocks.0.self_attn.q.weight"))
    before = gen_sd[probe_key].float().clone()
    result = pipeline.generator.load_state_dict(ema, strict=True)
    after = pipeline.generator.state_dict()[probe_key].float()
    load_report = {
        "checkpoint_bytes": ckpt.stat().st_size,  # sha256 is checked by the doctor (verify_sha256)
        "ema_tensors": len(ema),
        "generator_tensors": len(gen_sd),
        "missing_keys": list(result.missing_keys),
        "unexpected_keys": list(result.unexpected_keys),
        "probe_tensor": probe_key,
        "probe_changed_from_base": bool(not torch.equal(before, after)),
        "probe_equals_ema": bool(torch.equal(after, ema[probe_key].float())),
    }
    if not (load_report["probe_changed_from_base"] and load_report["probe_equals_ema"]):
        raise RuntimeError("EMA weights were not applied to the generator")
    del state, ema
    pipeline = pipeline.to(dtype=torch.bfloat16)
    offload = bool(job.get("offload_text_encoder", False))
    if offload:
        from utils.memory import DynamicSwapInstaller, gpu

        DynamicSwapInstaller.install_model(pipeline.text_encoder, device=gpu)
    else:
        pipeline.text_encoder.to(device=device)
    pipeline.generator.to(device=device)
    pipeline.vae.to(device=device)
    torch.cuda.synchronize()
    t_load = time.time() - t0
    _log("loaded", seconds=round(t_load, 2))

    # --- phase timers around text encoder, generator, VAE decode ---
    phase = {"text_encode": 0.0, "generator": 0.0, "generator_denoise": 0.0, "generator_context": 0.0, "vae_decode": 0.0}
    fwd = {"denoise": 0, "context": 0}

    current = {"video": 0}

    def _timed(fn, key, counter=None):
        def inner(*a, **k):
            if counter is not None:
                ts = k.get("timestep")
                kind = "context" if ts is not None and int(ts.flatten()[0]) == int(config.context_noise) else "denoise"
                fwd[kind] += 1
            torch.cuda.synchronize()
            s = time.time()
            out = fn(*a, **k)
            torch.cuda.synchronize()
            dt = time.time() - s
            phase[key] += dt
            if counter is not None:
                phase[f"generator_{kind}"] += dt
                _log("progress", video=current["video"], forwards=fwd["denoise"] + fwd["context"], forwards_per_video=FORWARDS_PER_VIDEO)
            return out

        return inner

    pipeline.text_encoder.forward = _timed(pipeline.text_encoder.forward, "text_encode")
    pipeline.generator.forward = _timed(pipeline.generator.forward, "generator", counter=True)
    vae_decoder_call = pipeline.vae.decoder
    pipeline.vae.decoder = _timed(vae_decoder_call, "vae_decode")

    out_dir = job_dir / "videos"
    out_dir.mkdir(exist_ok=False)
    records = []
    for idx, video in enumerate(job["videos"]):
        current["video"] = idx
        for k in phase:
            phase[k] = 0.0
        for k in fwd:
            fwd[k] = 0
        for k in counts:
            counts[k] = 0
        torch.cuda.reset_peak_memory_stats()
        torch.manual_seed(video["seed"])  # initial noise and the per-step re-noise draws
        noise = torch.randn([1, LATENT_FRAMES, *LATENT_SHAPE], device=device, dtype=torch.bfloat16)
        noise_sha = hashlib.sha256(noise.float().cpu().numpy().tobytes()).hexdigest()
        torch.cuda.synchronize()
        s = time.time()
        # low_memory mirrors upstream inference.py, which enables it (with the swap installer above) below 40 GB free VRAM
        frames, latents = pipeline.inference(noise=noise, text_prompts=[video["prompt"]], return_latents=True, low_memory=offload)
        torch.cuda.synchronize()
        t_inf = time.time() - s
        pipeline.vae.model.clear_cache()
        rgb = (frames[0].clamp(0, 1) * 255.0).round().to(torch.uint8).permute(0, 2, 3, 1).contiguous().cpu().numpy()
        if not (frames.isfinite().all() and latents.isfinite().all()):
            raise RuntimeError(f"video {idx}: non-finite output")
        _check_dispatch(job["profile"], counts, fwd, len(pipeline.generator.model.blocks))
        s = time.time()
        mp4 = out_dir / f"{idx:02d}.mp4"
        with av.open(str(mp4), "w") as container:
            stream = container.add_stream("libx264", rate=job.get("fps", 16))
            stream.width, stream.height, stream.pix_fmt = rgb.shape[2], rgb.shape[1], "yuv420p"
            stream.options = {"crf": "18"}
            for f in rgb:
                for packet in stream.encode(av.VideoFrame.from_ndarray(f, format="rgb24")):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)
        t_save = time.time() - s
        rec = {
            "index": idx,
            "file": f"videos/{mp4.name}",
            "sha256": _sha256(mp4),
            "seed": video["seed"],
            "prompt_sha256": hashlib.sha256(video["prompt"].encode("utf-8")).hexdigest(),
            "initial_noise_sha256": noise_sha,
            "latent_frames": int(latents.shape[1]),
            "rgb_frames": int(rgb.shape[0]),
            "height": int(rgb.shape[1]),
            "width": int(rgb.shape[2]),
            "fps": job.get("fps", 16),
            "seconds": {
                "inference_total": round(t_inf, 3),
                "text_encode": round(phase["text_encode"], 3),
                "generator_forwards": round(phase["generator"], 3),
                "generator_denoise": round(phase["generator_denoise"], 3),  # 28 forwards (4 steps x 7 blocks)
                "generator_context": round(phase["generator_context"], 3),  # 7 clean-context KV updates
                "vae_decode": round(phase["vae_decode"], 3),
                "other_in_inference": round(t_inf - phase["text_encode"] - phase["generator"] - phase["vae_decode"], 3),
                "mp4_write": round(t_save, 3),
            },
            "generator_forwards": dict(fwd),
            "attention_dispatch": dict(counts),
            "cuda_max_allocated_bytes": torch.cuda.max_memory_allocated(),
            "cuda_max_reserved_bytes": torch.cuda.max_memory_reserved(),
            "device_used_bytes_after": int(torch.cuda.mem_get_info()[1] - torch.cuda.mem_get_info()[0]),  # whole GPU, all processes
        }
        records.append(rec)
        _log("video_done", index=idx, seconds=rec["seconds"], dispatch=rec["attention_dispatch"], forwards=rec["generator_forwards"])

    result = {
        "schema_version": SCHEMA_VERSION,
        "job_id": job["job_id"],
        "profile": job["profile"],
        "effective_config": effective,
        "weights": load_report,
        "versions": versions,
        "model_load_seconds": round(t_load, 3),
        "process_seconds": round(time.time() - t_proc, 3),
        "host_peak_rss_bytes": _peak_rss_bytes(),
        "videos": records,
    }
    tmp = job_dir / "result.json.tmp"
    tmp.write_text(json.dumps(result, indent=1, ensure_ascii=False), encoding="utf-8")
    tmp.replace(job_dir / "result.json")
    _log("done", job_id=job["job_id"])
    return 0


if __name__ == "__main__":
    code = main(sys.argv)
    sys.stdout.flush()
    sys.stderr.flush()
    _stop_own_group()  # no compile workers or other helpers outlive the job
    os._exit(code)  # do not wait for the watcher threads (or CUDA teardown) on the way out
