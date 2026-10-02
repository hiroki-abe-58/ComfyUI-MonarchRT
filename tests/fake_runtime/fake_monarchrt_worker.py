"""Entry point used by the CPU tests: the real runtime/monarchrt_worker.py with a fake engine (no torch, no GPU).

The worker manager copies this file, the real worker and job modules and a fixture MP4 into the
session's code folder and starts it exactly like the real worker. The fake engine behaves according to
the first word of the prompt:

  MODE:ok        35 progress steps, then the fixture MP4 (81 frames, 832x480, 16 fps)
  MODE:hang      progress, then waits until cancelled (cooperative cancel)
  MODE:hardhang  sleeps and ignores cancel (forces the manager to stop the worker)
  MODE:fail      raises a normal error (the worker stays)
  MODE:cudafail  raises an error that mentions CUDA (fatal: the worker is replaced)
  MODE:crash     the worker process exits immediately
"""

from __future__ import annotations

import hashlib
import os
import shutil
import sys
import time
import uuid
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import monarchrt_job as job_mod  # noqa: E402
import monarchrt_worker  # noqa: E402

ENGINE_ID = uuid.uuid4().hex[:12]


class FakeEngine:
    loads = 0

    def __init__(self, upstream, models, ckpt, profile, offload, on_progress=None):
        FakeEngine.loads += 1
        self.profile = profile
        self.load_seconds = 0.05
        self.load_report = {"fake": True}
        self.effective = {"profile": profile}
        self.videos_generated = 0
        self.should_cancel = None
        self.on_progress = on_progress

    def versions(self):
        return {"fake": True}

    def set_profile(self, profile):
        self.profile = profile
        self.effective = {"profile": profile}
        return self.effective

    def _step(self, index, k, delay):
        if self.should_cancel is not None and self.should_cancel():
            raise job_mod.JobCancelled("cancelled")
        if self.on_progress is not None:
            self.on_progress(index, k)
        time.sleep(delay)

    def generate(self, index, video, out_dir, fps):
        mode = video["prompt"].split()[0]
        if mode == "MODE:crash":
            os._exit(9)
        if mode == "MODE:fail":
            raise RuntimeError("simulated failure")
        if mode == "MODE:cudafail":
            raise RuntimeError("CUDA error: simulated device fault")
        for k in range(1, 36):
            self._step(index, k, 0.005)
            if mode == "MODE:hang" and k == 3:
                while True:
                    self._step(index, k, 0.05)
            if mode == "MODE:hardhang" and k == 3:
                time.sleep(3600)
        mp4 = out_dir / f"{index:02d}.mp4"
        shutil.copyfile(HERE / "fixture.mp4", mp4)
        self.videos_generated += 1
        return {
            "index": index,
            "file": f"videos/{mp4.name}",
            "sha256": hashlib.sha256(mp4.read_bytes()).hexdigest(),
            "seed": video["seed"],
            "rgb_frames": 81,
            "height": 480,
            "width": 832,
            "fps": fps,
            "seconds": {"inference_total": 0.2},
            "generator_forwards": {"denoise": 28, "context": 7},
            "attention_dispatch": {"fake": True, "profile": self.profile},
            "engine_id": ENGINE_ID,
            "engine_loads": FakeEngine.loads,
            "worker_pid": os.getpid(),
        }


if __name__ == "__main__":
    code = monarchrt_worker.main(sys.argv, engine_factory=FakeEngine)
    sys.stderr.flush()
    job_mod._stop_own_group()
    os._exit(code)
