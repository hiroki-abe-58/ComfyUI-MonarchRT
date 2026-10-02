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


def apply_env(env: dict) -> None:
    """Drop anything that looks like a credential, apply the allowlisted runtime settings, force offline mode."""
    for key in list(os.environ):
        if SECRET_RE.search(key):
            del os.environ[key]
    os.environ.update(env)
    os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    # Everything is loaded from local files; never reach out to the Hugging Face Hub.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"


def become_session_leader(pid_file: Path) -> None:
    """Own process group, so a parent can stop the whole tree; record pid, pgid and start time."""
    if hasattr(os, "setsid"):
        try:
            os.setsid()
        except OSError:
            pass
    me = _proc_stat(os.getpid())
    record = {"pid": os.getpid(), "pgid": os.getpgid(0) if hasattr(os, "getpgid") else None, "starttime": int(me[19]) if me else None}
    pid_file.write_text(json.dumps(record), encoding="utf-8")


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

    apply_env(job.get("env", {}))
    become_session_leader(job_dir / "runner.pid")
    _install_cancel_watch(job_dir, watch_stdin)
    _log("start", job_id=job["job_id"], profile=job["profile"], videos=len(job["videos"]))

    try:
        return _run(job, job_dir)
    except Exception as exc:  # report and fail; never write a result on failure
        import traceback

        _log("failed", error_type=type(exc).__name__, error=str(exc)[:2000], traceback=traceback.format_exc()[-4000:])
        return 3


class JobCancelled(RuntimeError):
    """Raised between generator forwards when the caller asked to stop."""


# Process-wide state of the wrappers installed by Engine (installed once per process, never stacked).
_COUNTS = {
    "monarch_kv_calls": 0,  # causal self-attention with KV cache, Monarch branch
    "monarch_triton_calls": 0,  # ... of which reached the fused Triton kernel (num_iters == 1)
    "monarch_torch_slow_calls": 0,  # ... of which used the pure-torch iterative path (num_iters > 1)
    "monarch_no_cache_calls": 0,  # Monarch without KV cache (training path; expected 0)
    "dense_self_attn_calls": 0,  # causal self-attention with KV cache, dense branch
    "flex_attn_calls": 0,  # dense without KV cache (training path; expected 0)
    "cross_attn_calls": 0,  # text cross-attention (always dense flash-attn)
}
_AUTOTUNE = {"bench_calls": 0}  # Triton autotuner benchmark runs (one per config per new tuning key)
_PATCHED = {"done": False}


def profile_config(upstream: Path, profile: str):
    """Merged upstream config for a profile, with the Monarch settings applied and asserted."""
    from omegaconf import OmegaConf

    cfg_file, overrides = PROFILES[profile]
    config = OmegaConf.merge(OmegaConf.load(upstream / "configs" / "default_config.yaml"), OmegaConf.load(upstream / "configs" / cfg_file))
    if overrides is not None:
        for k, v in overrides.items():
            config.monarch_args[k] = v
    margs = OmegaConf.to_container(config.get("monarch_args", {}), resolve=True) or {}
    expected = overrides or {"enable": False}
    for k, v in expected.items():
        if margs.get(k) != v:
            raise RuntimeError(f"monarch_args.{k} = {margs.get(k)!r}, expected {v!r}")
    return config, cfg_file, margs


class Engine:
    """The part of a job that is done once per process: guards, wrappers, pipeline, weights, device placement.

    Videos are generated one at a time with ``generate``; everything that belongs to a single video (seed,
    initial noise, KV / cross-attention caches, VAE cache, counters, timers) is reset for each call. The
    attention profile can be switched between videos with ``set_profile``: the upstream model applies
    ``monarch_args`` through a property that only sets per-block attributes, and the two upstream configs
    differ in nothing else.
    """

    def __init__(self, upstream: Path, models: Path, ckpt: Path, profile: str, offload: bool, on_progress=None):
        self.upstream, self.models, self.ckpt = upstream, models, ckpt
        for p, what in (
            (upstream / "pipeline" / "causal_inference.py", "upstream checkout"),
            (models / "wan_models" / "Wan2.1-T2V-1.3B" / "config.json", "Wan2.1-T2V-1.3B"),
            (ckpt, "checkpoint"),
        ):
            if not p.is_file():
                raise FileNotFoundError(f"{what} not found")
        if str(upstream) not in sys.path:
            sys.path.insert(0, str(upstream))
        os.chdir(models)  # upstream resolves wan_models/... relative to the working directory
        self.on_progress = on_progress
        self.should_cancel = None
        self._install_patches()

        import torch
        from pipeline import CausalInferencePipeline

        self.torch = torch
        config, cfg_file, margs = profile_config(upstream, profile)
        self.config = config
        self.profile = profile
        self.offload = bool(offload)
        self.effective = self._effective(config, cfg_file, margs)

        # --- model load ---
        torch.set_grad_enabled(False)
        device = torch.device("cuda")
        self.device = device
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
        self.load_report = {
            "checkpoint_bytes": ckpt.stat().st_size,  # sha256 is checked by the doctor (verify_sha256)
            "ema_tensors": len(ema),
            "generator_tensors": len(gen_sd),
            "missing_keys": list(result.missing_keys),
            "unexpected_keys": list(result.unexpected_keys),
            "probe_tensor": probe_key,
            "probe_changed_from_base": bool(not torch.equal(before, after)),
            "probe_equals_ema": bool(torch.equal(after, ema[probe_key].float())),
        }
        if not (self.load_report["probe_changed_from_base"] and self.load_report["probe_equals_ema"]):
            raise RuntimeError("EMA weights were not applied to the generator")
        del state, ema
        pipeline = pipeline.to(dtype=torch.bfloat16)
        if self.offload:
            from utils.memory import DynamicSwapInstaller, gpu

            DynamicSwapInstaller.install_model(pipeline.text_encoder, device=gpu)
        else:
            pipeline.text_encoder.to(device=device)
        pipeline.generator.to(device=device)
        pipeline.vae.to(device=device)
        torch.cuda.synchronize()
        self.load_seconds = time.time() - t0
        self.pipeline = pipeline
        self.layers = len(pipeline.generator.model.blocks)

        # --- phase timers around text encoder, generator, VAE decode (instance attributes of this pipeline) ---
        self.phase = {"text_encode": 0.0, "generator": 0.0, "generator_denoise": 0.0, "generator_context": 0.0, "vae_decode": 0.0}
        self.fwd = {"denoise": 0, "context": 0}
        self.current = {"video": 0}
        pipeline.text_encoder.forward = self._timed(pipeline.text_encoder.forward, "text_encode")
        pipeline.generator.forward = self._timed(pipeline.generator.forward, "generator", counter=True)
        pipeline.vae.decoder = self._timed(pipeline.vae.decoder, "vae_decode")
        self.videos_generated = 0

    # -- process-wide patches (idempotent) -----------------------------------------------------------------
    @staticmethod
    def _install_patches() -> None:
        if _PATCHED["done"]:
            return
        import torch

        _orig_load = torch.load

        def _safe_load(*args, **kwargs):
            kwargs["weights_only"] = True  # upstream passes weights_only=False for T5; refuse pickled code objects
            kwargs.setdefault("mmap", True)  # page-cache backed instead of a second full copy in RAM
            return _orig_load(*args, **kwargs)

        torch.load = _safe_load
        import utils.wan_wrapper as wan_wrapper
        import wan.modules.attention as wan_attention
        import wan.modules.causal_model as causal_model
        import wan.modules.model as wan_model
        import wan.modules.monarch_attn as monarch_attn

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

        def _wrap(fn, key):
            def inner(*a, **k):
                _COUNTS[key] += 1
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
        if not (wan_attention.FLASH_ATTN_2_AVAILABLE or wan_attention.FLASH_ATTN_3_AVAILABLE):
            raise RuntimeError("flash-attn is required by the upstream cross-attention (wan/modules/attention.py)")

        from triton.runtime import autotuner

        _orig_bench = autotuner.Autotuner._bench

        def _bench(self, *a, **k):
            _AUTOTUNE["bench_calls"] += 1
            return _orig_bench(self, *a, **k)

        autotuner.Autotuner._bench = _bench
        _PATCHED["done"] = True

    def versions(self) -> dict:
        import flash_attn
        import flashinfer
        import triton

        torch = self.torch
        return {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "triton": triton.__version__,
            "device": torch.cuda.get_device_name(0),
            "capability": ".".join(map(str, torch.cuda.get_device_capability(0))),
            "flashinfer": getattr(flashinfer, "__version__", "unknown"),
            "flash_attn": flash_attn.__version__,
        }

    def _effective(self, config, cfg_file: str, margs: dict) -> dict:
        return {
            "config_file": cfg_file,
            "monarch_args": margs,
            "denoising_step_list": list(config.denoising_step_list),
            "warp_denoising_step": bool(config.warp_denoising_step),
            "num_frame_per_block": int(config.num_frame_per_block),
            "context_noise": int(config.context_noise),
            "timestep_shift": float(config.model_kwargs.timestep_shift),
            "offload_text_encoder": self.offload,
        }

    def set_profile(self, profile: str) -> dict:
        """Switch the attention profile between videos (no reload)."""
        if profile != self.profile:
            config, cfg_file, margs = profile_config(self.upstream, profile)
            for key in ("denoising_step_list", "warp_denoising_step", "num_frame_per_block", "context_noise"):
                if config.get(key) != self.config.get(key):
                    raise RuntimeError(f"profile {profile!r} changes {key}; it needs a new process")
            self.pipeline.generator.model.monarch_args = margs  # upstream property: sets every block's self-attention
            self.profile, self.effective = profile, self._effective(config, cfg_file, margs)
        return self.effective

    def _timed(self, fn, key, counter=False):
        torch = self.torch

        def inner(*a, **k):
            if self.should_cancel is not None and self.should_cancel():
                raise JobCancelled(f"cancelled before {key}")
            if counter:
                ts = k.get("timestep")
                kind = "context" if ts is not None and int(ts.flatten()[0]) == int(self.config.context_noise) else "denoise"
                self.fwd[kind] += 1
            torch.cuda.synchronize()
            s = time.time()
            out = fn(*a, **k)
            torch.cuda.synchronize()
            dt = time.time() - s
            self.phase[key] += dt
            if counter:
                self.phase[f"generator_{kind}"] += dt
                if self.on_progress is not None:
                    self.on_progress(self.current["video"], self.fwd["denoise"] + self.fwd["context"])
            return out

        return inner

    def generate(self, index: int, video: dict, out_dir: Path, fps: int) -> dict:
        """Generate one video into out_dir/NN.mp4 and return its record."""
        import av

        torch = self.torch
        pipeline = self.pipeline
        self.current["video"] = index
        for k in self.phase:
            self.phase[k] = 0.0
        for k in self.fwd:
            self.fwd[k] = 0
        for k in _COUNTS:
            _COUNTS[k] = 0
        bench_before = _AUTOTUNE["bench_calls"]
        torch.cuda.reset_peak_memory_stats()
        torch.manual_seed(video["seed"])  # initial noise and the per-step re-noise draws
        noise = torch.randn([1, LATENT_FRAMES, *LATENT_SHAPE], device=self.device, dtype=torch.bfloat16)
        noise_sha = hashlib.sha256(noise.float().cpu().numpy().tobytes()).hexdigest()
        torch.cuda.synchronize()
        s = time.time()
        try:
            # low_memory mirrors upstream inference.py, which enables it (with the swap installer above) below 40 GB free VRAM
            frames, latents = pipeline.inference(noise=noise, text_prompts=[video["prompt"]], return_latents=True, low_memory=self.offload)
        finally:
            pipeline.vae.model.clear_cache()
        torch.cuda.synchronize()
        t_inf = time.time() - s
        rgb = (frames[0].clamp(0, 1) * 255.0).round().to(torch.uint8).permute(0, 2, 3, 1).contiguous().cpu().numpy()
        if not (frames.isfinite().all() and latents.isfinite().all()):
            raise RuntimeError(f"video {index}: non-finite output")
        _check_dispatch(self.profile, _COUNTS, self.fwd, self.layers)
        latent_sha = hashlib.sha256(latents.float().cpu().numpy().tobytes()).hexdigest()
        frames_sha = hashlib.sha256(rgb.tobytes()).hexdigest()
        del frames, latents
        s = time.time()
        mp4 = out_dir / f"{index:02d}.mp4"
        with av.open(str(mp4), "w") as container:
            stream = container.add_stream("libx264", rate=fps)
            stream.width, stream.height, stream.pix_fmt = rgb.shape[2], rgb.shape[1], "yuv420p"
            stream.options = {"crf": "18"}
            for f in rgb:
                for packet in stream.encode(av.VideoFrame.from_ndarray(f, format="rgb24")):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)
        t_save = time.time() - s
        phase = self.phase
        self.videos_generated += 1
        return {
            "index": index,
            "file": f"videos/{mp4.name}",
            "sha256": _sha256(mp4),
            "seed": video["seed"],
            "prompt_sha256": hashlib.sha256(video["prompt"].encode("utf-8")).hexdigest(),
            "initial_noise_sha256": noise_sha,
            "latents_sha256": latent_sha,  # model output before VAE decode (bf16 -> fp32 bytes)
            "rgb_frames_sha256": frames_sha,  # decoded uint8 frames before H.264 encoding
            "latent_frames": LATENT_FRAMES,
            "rgb_frames": int(rgb.shape[0]),
            "height": int(rgb.shape[1]),
            "width": int(rgb.shape[2]),
            "fps": fps,
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
            "generator_forwards": dict(self.fwd),
            "attention_dispatch": dict(_COUNTS),
            "autotune_bench_calls": _AUTOTUNE["bench_calls"] - bench_before,
            "cuda_max_allocated_bytes": torch.cuda.max_memory_allocated(),
            "cuda_max_reserved_bytes": torch.cuda.max_memory_reserved(),
            "cuda_allocated_bytes_after": torch.cuda.memory_allocated(),
            "device_used_bytes_after": int(torch.cuda.mem_get_info()[1] - torch.cuda.mem_get_info()[0]),  # whole GPU, all processes
        }


def write_result(job_dir: Path, result: dict) -> None:
    tmp = job_dir / "result.json.tmp"
    tmp.write_text(json.dumps(result, indent=1, ensure_ascii=False), encoding="utf-8")
    tmp.replace(job_dir / "result.json")


def _run(job: dict, job_dir: Path) -> int:
    t_proc = time.time()
    engine = Engine(
        Path(job["upstream_dir"]).resolve(),
        Path(job["models_dir"]).resolve(),
        Path(job["checkpoint"]).resolve(),
        job["profile"],
        bool(job.get("offload_text_encoder", False)),
        on_progress=lambda video, forwards: _log("progress", video=video, forwards=forwards, forwards_per_video=FORWARDS_PER_VIDEO),
    )
    _log("loaded", seconds=round(engine.load_seconds, 2))
    out_dir = job_dir / "videos"
    out_dir.mkdir(exist_ok=False)
    records = []
    for idx, video in enumerate(job["videos"]):
        rec = engine.generate(idx, video, out_dir, job.get("fps", 16))
        records.append(rec)
        _log("video_done", index=idx, seconds=rec["seconds"], dispatch=rec["attention_dispatch"], forwards=rec["generator_forwards"])
    write_result(
        job_dir,
        {
            "schema_version": SCHEMA_VERSION,
            "job_id": job["job_id"],
            "profile": job["profile"],
            "backend": "one-shot",
            "effective_config": engine.effective,
            "weights": engine.load_report,
            "versions": engine.versions(),
            "model_load_seconds": round(engine.load_seconds, 3),
            "process_seconds": round(time.time() - t_proc, 3),
            "host_peak_rss_bytes": _peak_rss_bytes(),
            "videos": records,
        },
    )
    _log("done", job_id=job["job_id"])
    return 0


if __name__ == "__main__":
    code = main(sys.argv)
    sys.stdout.flush()
    sys.stderr.flush()
    _stop_own_group()  # no compile workers or other helpers outlive the job
    os._exit(code)  # do not wait for the watcher threads (or CUDA teardown) on the way out
