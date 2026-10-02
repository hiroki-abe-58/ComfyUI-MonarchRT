"""ComfyUI nodes. Importing this module loads no CUDA, no model and no upstream code."""

from __future__ import annotations

import json
import time

from . import client
from .config import CONFIG_ENV, CONFIG_FILENAME, ConfigError, config_path, load_runtimes

NO_RUNTIME = "(no runtime configured)"
PROFILE_HELP = {
    "monarch_h2": "MonarchRT, training-free: Monarch attention on the public dense Self-Forcing weights, h_reduce=2 (about 90% effective attention sparsity; the training-free setting discussed in upstream issue #2).",
    "monarch_h1": "MonarchRT, training-free, h_reduce=1 (the upstream config default, about 95% effective sparsity; the paper pairs this with trained weights, so expect lower quality without them).",
    "dense": "Baseline: the same Self-Forcing weights with dense attention (flash-attn). Use it to compare against MonarchRT.",
}


def _runtime_ids() -> list[str]:
    try:
        ids = sorted(load_runtimes())
    except ConfigError:
        ids = []
    return ids or [NO_RUNTIME]


def _resolve(runtime_id: str) -> client.Runtime:
    try:
        runtimes = load_runtimes()
    except ConfigError as exc:
        raise ValueError(f"MonarchRT runtime config is invalid: {exc}") from exc
    if runtime_id not in runtimes:
        where = config_path()
        raise ValueError(
            f"MonarchRT runtime {runtime_id!r} is not configured. An administrator registers runtimes in "
            f"{where if where else CONFIG_FILENAME} (or the file named by {CONFIG_ENV}); see the README."
        )
    return runtimes[runtime_id]


class MonarchRTRuntime:
    """Select an administrator-registered runtime (WSL2 / Linux environment with the upstream code and weights)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "runtime_id": (_runtime_ids(), {"tooltip": f"Runtimes come from {CONFIG_FILENAME} (administrator config), never from the workflow."}),
            }
        }

    RETURN_TYPES = ("MONARCHRT_RUNTIME",)
    RETURN_NAMES = ("runtime",)
    FUNCTION = "select"
    CATEGORY = "MonarchRT"
    DESCRIPTION = "Pick a MonarchRT runtime registered by the administrator."

    def select(self, runtime_id):
        rt = _resolve(runtime_id)
        return ({"runtime_id": rt.id},)


class MonarchRTGenerate:
    """Text-to-video with Self-Forcing (Wan2.1-T2V-1.3B), dense or MonarchRT attention, in the selected runtime."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "runtime": ("MONARCHRT_RUNTIME",),
                "prompt": ("STRING", {"multiline": True, "default": "", "tooltip": "Used as written (no prompt rewriting)."}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2**31 - 1, "control_after_generate": True}),
                "attention": (list(client.PROFILES), {"default": "monarch_h2", "tooltip": " | ".join(f"{k}: {v}" for k, v in PROFILE_HELP.items())}),
                "videos": (
                    "INT",
                    {
                        "default": 1,
                        "min": 1,
                        "max": 4,
                        "tooltip": "Videos from one runtime process (seeds seed, seed+1, ...): the model is loaded and the kernels are tuned once.",
                    },
                ),
            }
        }

    RETURN_TYPES = ("VIDEO", "STRING")
    RETURN_NAMES = ("video", "report")
    OUTPUT_IS_LIST = (True, False)
    FUNCTION = "generate"
    CATEGORY = "MonarchRT"
    DESCRIPTION = (
        "480x832, 81 frames at 16 fps. Runs the upstream Self-Forcing / MonarchRT pipeline in the runtime, returns real VIDEO outputs and a JSON report."
    )

    def generate(self, runtime, prompt, seed, attention, videos):
        import comfy.model_management as mm
        import comfy.utils
        from comfy_api.latest import InputImpl

        if not isinstance(runtime, dict) or "runtime_id" not in runtime:
            raise ValueError("connect a MonarchRT Runtime node")
        rt = _resolve(runtime["runtime_id"])
        if attention not in client.PROFILES:
            raise ValueError(f"attention must be one of {client.PROFILES}")
        if not prompt or not prompt.strip():
            raise ValueError("prompt is empty")
        specs = [{"prompt": prompt, "seed": int(seed) + i} for i in range(int(videos))]
        if specs[-1]["seed"] > 2**31 - 1:
            raise ValueError("seed + videos - 1 exceeds 2^31-1")
        total = client.FORWARDS_PER_VIDEO * len(specs)
        pbar = comfy.utils.ProgressBar(total)
        t0 = time.time()
        try:
            outcome = client.generate(
                rt,
                attention,
                specs,
                interrupted=mm.processing_interrupted,
                on_progress=lambda done, tot: pbar.update_absolute(done, tot),
            )
        except client.JobCancelled as exc:
            raise mm.InterruptProcessingException() from exc
        report = _report(outcome, rt, attention, time.time() - t0)
        return ([InputImpl.VideoFromFile(str(p)) for p in outcome.videos], json.dumps(report, indent=1, ensure_ascii=False))


def _report(outcome: client.JobOutcome, rt, profile: str, wall_s: float) -> dict:
    r = outcome.result
    return {
        "runtime_id": rt.id,
        "profile": profile,
        "profile_note": PROFILE_HELP[profile],
        "training_free": profile != "dense",
        "job": outcome.job_dir.name,
        "wall_seconds": round(wall_s, 2),
        "model_load_seconds": r.get("model_load_seconds"),
        "effective_config": r.get("effective_config"),
        "weights": r.get("weights"),
        "versions": r.get("versions"),
        "videos": [
            {
                k: v[k]
                for k in (
                    "seed",
                    "rgb_frames",
                    "width",
                    "height",
                    "fps",
                    "seconds",
                    "generator_forwards",
                    "attention_dispatch",
                    "cuda_max_allocated_bytes",
                    "sha256",
                )
                if k in v
            }
            for v in r.get("videos", [])
        ],
    }


class MonarchRTDoctor:
    """Check a runtime: upstream files at the pinned commit, model files, CUDA/Triton/flash-attn, optional kernel check."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "runtime_id": (_runtime_ids(),),
                "verify_sha256": ("BOOLEAN", {"default": False, "tooltip": "Hash every model file (about 23 GB of reads)."}),
                "kernel_check": (
                    "BOOLEAN",
                    {"default": False, "tooltip": "Compare the Monarch Triton kernel with its torch reference (first run tunes kernels, a few minutes)."},
                ),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("report",)
    FUNCTION = "check"
    CATEGORY = "MonarchRT"
    OUTPUT_NODE = True
    DESCRIPTION = "Diagnose a MonarchRT runtime without generating a video."

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return time.time()  # always re-check when queued

    def check(self, runtime_id, verify_sha256, kernel_check):
        import comfy.model_management as mm

        where = config_path()
        try:
            runtimes = load_runtimes()
        except ConfigError as exc:
            report = {"overall": "fail", "config": str(where), "error": str(exc)}
            return {"ui": {"text": [f"config error: {exc}"]}, "result": (json.dumps(report, indent=1),)}
        if runtime_id not in runtimes:
            report = {
                "overall": "fail",
                "config": str(where),
                "error": f"no runtime {runtime_id!r}; create {CONFIG_FILENAME} (see examples/ and the README)",
                "configured": sorted(runtimes),
            }
            return {"ui": {"text": [report["error"]]}, "result": (json.dumps(report, indent=1),)}
        try:
            report = client.doctor(runtimes[runtime_id], verify_sha256=verify_sha256, kernel_check=kernel_check, interrupted=mm.processing_interrupted)
        except client.JobCancelled as exc:
            raise mm.InterruptProcessingException() from exc
        lines = [f"{runtime_id}: {report.get('overall')}"] + [f"{c['status']:>4}  {c['name']}" for c in report.get("checks", [])]
        return {"ui": {"text": ["\n".join(lines)]}, "result": (json.dumps(report, indent=1, ensure_ascii=False),)}


NODE_CLASS_MAPPINGS = {
    "MonarchRTRuntime": MonarchRTRuntime,
    "MonarchRTGenerate": MonarchRTGenerate,
    "MonarchRTDoctor": MonarchRTDoctor,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "MonarchRTRuntime": "MonarchRT Runtime",
    "MonarchRTGenerate": "MonarchRT Generate (Self-Forcing T2V)",
    "MonarchRTDoctor": "MonarchRT Doctor",
}
