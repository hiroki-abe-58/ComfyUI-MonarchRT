"""End-to-end check of the persistent worker through ComfyUI's HTTP API (maintainer tool, real GPU runtime).

Every step is a separate ComfyUI queue job. The node cache is cleared (POST /free) before jobs that repeat
inputs, so a repeated seed really runs again; reuse is proven with the worker pid, the per-job model-load
and Triton-autotune counters from the worker, and the output hashes - not with cached node outputs.

Usage:
  python scripts/gpu_e2e_persistent.py --comfyui DIR --python EXE --config RUNTIMES.json --runtime ID
         --out report.json [--reference refs.json]

RUNTIMES.json must contain ID (a 'wsl' runtime). The script derives extra runtimes from it (one-shot,
short idle timeout, 1-minute timeout, missing checkpoint). --reference maps "profile|prompt|seed" to the
sha256 of the one-shot MP4 for the same inputs. Only processes started by this script are stopped.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gpu_e2e as base  # noqa: E402

PROMPTS = [
    "A red fox trotting through fresh snow in a quiet pine forest at sunrise, warm golden light filtering between the trees, the camera tracking alongside at ground level.",
    "Ocean waves crashing against dark volcanic rocks on a stormy coastline, white spray bursting into the air, overcast sky, cinematic wide shot.",
    "A rainy city street at night lit by neon signs, people with umbrellas walking past glowing shop windows, colorful reflections shimmering on the wet pavement.",
]


def generate_prompt(runtime_id, profile, prompt, seed, videos=1, backend="runtime default", prefix="persist"):
    return {
        "1": {"class_type": "MonarchRTRuntime", "inputs": {"runtime_id": runtime_id, "backend": backend}},
        "2": {"class_type": "MonarchRTGenerate", "inputs": {"runtime": ["1", 0], "prompt": prompt, "seed": seed, "attention": profile, "videos": videos}},
        "3": {"class_type": "SaveVideo", "inputs": {"video": ["2", 0], "filename_prefix": f"{prefix}/{profile}", "format": "mp4", "format.codec": "h264"}},
        "4": {"class_type": "PreviewAny", "inputs": {"source": ["2", 1]}},
    }


def worker_prompt(action):
    return {
        "1": {"class_type": "MonarchRTWorker", "inputs": {"action": action}},
        "2": {"class_type": "PreviewAny", "inputs": {"source": ["1", 0]}},
    }


class Runner:
    def __init__(self, args, cfg_path, work):
        self.args, self.cfg_path, self.work = args, cfg_path, work
        self.comfy = None
        self.records = []

    def start_comfy(self):
        self.comfy = base.Comfy(self.args.comfyui, self.args.python, self.cfg_path, self.work)
        return self.comfy

    def kill_comfy_hard(self):
        """Simulate a ComfyUI crash: kill its process tree (no atexit, no unload)."""
        subprocess.run(["taskkill", "/T", "/F", "/PID", str(self.comfy.proc.pid)], capture_output=True)
        self.comfy.proc.wait(timeout=30)
        self.comfy.log.close()

    def free_cache(self):
        base.http(self.comfy.base, "/free", {"free_memory": True})
        time.sleep(1)

    def run(self, label, prompt, timeout=3600, free=False, expect="success"):
        if free:
            self.free_cache()
        t0 = time.time()
        pid = self.comfy.queue(prompt)
        h = self.comfy.wait(pid, timeout)
        wall = round(time.time() - t0, 2)
        outs = h.get("outputs", {})
        rep = None
        for node in outs.values():
            for txt in node.get("text", []):
                try:
                    rep = json.loads(txt)
                except (TypeError, json.JSONDecodeError):
                    continue
        msgs = [m[0] for m in h["status"].get("messages", [])]
        err = next((m[1].get("exception_message", "")[:400] for m in h["status"].get("messages", []) if m[0] == "execution_error"), None)
        rec = {"label": label, "status": h["status"].get("status_str"), "messages": msgs, "error": err, "wall_s": wall, "expect": expect}
        if rep and "videos" in rep:
            w = rep.get("worker") or {}
            rec.update(
                backend=rep.get("backend"),
                profile=rep.get("profile"),
                worker_id=w.get("worker_id"),
                worker_pid=w.get("pid"),
                started_for_this_job=w.get("started_for_this_job"),
                job_number_in_worker=w.get("job_number_in_worker"),
                autotune_bench_calls_this_job=w.get("autotune_bench_calls_this_job"),
                worker_model_load_seconds=w.get("worker_model_load_seconds"),
                model_load_seconds=rep.get("model_load_seconds"),
                rss_bytes_after=w.get("rss_bytes_after"),
                videos=[
                    {
                        k: v.get(k)
                        for k in ("seed", "sha256", "latents_sha256", "rgb_frames_sha256", "seconds", "autotune_bench_calls", "cuda_max_reserved_bytes")
                    }
                    for v in rep["videos"]
                ],
                report_wall_seconds=rep.get("wall_seconds"),
            )
        if rep and "action" in rep:
            rec["worker_report"] = rep
        rec["gpu_used_mib_after"] = base.gpu_used_mib()
        self.records.append(rec)
        print(
            json.dumps({k: rec.get(k) for k in ("label", "status", "wall_s", "worker_pid", "started_for_this_job", "autotune_bench_calls_this_job", "error")}),
            flush=True,
        )
        return rec


def wsl_procs(distro, needle):
    return base.wsl_processes(distro, needle)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--comfyui", required=True, type=Path)
    ap.add_argument("--python", required=True)
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--runtime", required=True)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--reference", type=Path)
    args = ap.parse_args()
    work = args.out.parent / f"persist-{time.strftime('%Y%m%d-%H%M%S')}"
    work.mkdir(parents=True)
    rt = json.loads(args.config.read_text(encoding="utf-8"))["runtimes"][args.runtime]
    distro = rt["distro"]
    rid = args.runtime
    cfg = {
        "schema_version": 1,
        "runtimes": {
            rid: {**rt, "backend": "persistent", "worker_idle_seconds": 600},
            "oneshot": {**rt, "backend": "one-shot"},
            "idle60": {**rt, "backend": "persistent", "worker_idle_seconds": 60},
            "timeout1": {**rt, "backend": "persistent", "timeout_minutes": 1},
            "missingckpt": {**rt, "backend": "persistent", "checkpoint": rt["checkpoint"] + ".missing"},
        },
    }
    cfg_path = work / "monarchrt.runtimes.json"
    cfg_path.write_text(json.dumps(cfg, indent=1), encoding="utf-8")
    refs = json.loads(args.reference.read_text(encoding="utf-8")) if args.reference else {}
    r = Runner(args, cfg_path, work)
    report = {"gpu_idle_mib_before": base.gpu_used_mib(), "checks": {}}
    P1, P2, P3 = PROMPTS
    try:
        r.start_comfy()
        report["port"] = r.comfy.port
        # 1) three (and more) separate queue jobs on one warm worker
        j1 = r.run("J1 monarch_h2 P1 s101 (cold)", generate_prompt(rid, "monarch_h2", P1, 101))
        j2 = r.run("J2 monarch_h2 P2 s101 (warm)", generate_prompt(rid, "monarch_h2", P2, 101))
        j3 = r.run("J3 monarch_h2 P1 s101 again (same seed, node cache cleared)", generate_prompt(rid, "monarch_h2", P1, 101), free=True)
        j4 = r.run("J4 dense P1 s101 (profile switch)", generate_prompt(rid, "dense", P1, 101))
        j5 = r.run("J5 monarch_h2 P3 s2026 (dense -> Monarch)", generate_prompt(rid, "monarch_h2", P3, 2026))
        j6 = r.run("J6 dense P1 s101 again (Monarch -> dense)", generate_prompt(rid, "dense", P1, 101), free=True)
        j7 = r.run("J7 monarch_h1 P2 s2026 (new profile in warm worker)", generate_prompt(rid, "monarch_h1", P2, 2026))
        j8 = r.run("J8 monarch_h1 P2 s2026 again", generate_prompt(rid, "monarch_h1", P2, 2026), free=True)
        warm = [j1, j2, j3, j4, j5, j6, j7, j8]
        report["checks"]["same_worker_for_8_jobs"] = len({j.get("worker_pid") for j in warm}) == 1 and all(j["status"] == "success" for j in warm)
        report["checks"]["only_first_job_started_worker"] = [j.get("started_for_this_job") for j in warm] == [True] + [False] * 7
        report["checks"]["job_numbers"] = [j.get("job_number_in_worker") for j in warm]
        report["checks"]["autotune_bench_calls_per_job"] = [j.get("autotune_bench_calls_this_job") for j in warm]
        report["checks"]["same_seed_rerun_identical"] = j1["videos"][0]["sha256"] == j3["videos"][0]["sha256"]
        report["checks"]["dense_monarch_dense_identical"] = j4["videos"][0]["sha256"] == j6["videos"][0]["sha256"]
        report["checks"]["h1_rerun_identical"] = j7["videos"][0]["sha256"] == j8["videos"][0]["sha256"]
        report["checks"]["memory_after_each_job"] = [
            {"label": j["label"][:3], "rss_gib": round((j.get("rss_bytes_after") or 0) / 2**30, 2), "gpu_used_mib": j["gpu_used_mib_after"]} for j in warm
        ]
        for j, key in ((j1, "monarch_h2|P1|101"), (j4, "dense|P1|101"), (j5, "monarch_h2|P3|2026"), (j7, "monarch_h1|P2|2026")):
            if key in refs:
                report["checks"][f"matches_one_shot_{key}"] = j["videos"][0]["sha256"] == refs[key]
        # 2) status, back-to-back queueing, then unload
        st = r.run("status", worker_prompt("status"))
        report["checks"]["status_alive"] = bool(st.get("worker_report", {}).get("worker", {}).get("alive"))
        q1 = r.comfy.queue(generate_prompt(rid, "monarch_h2", P2, 7))
        q2 = r.comfy.queue(generate_prompt(rid, "monarch_h2", P3, 8))
        h1, h2 = r.comfy.wait(q1, 900), r.comfy.wait(q2, 900)
        report["checks"]["back_to_back_queue"] = [h1["status"]["status_str"], h2["status"]["status_str"]]
        un = r.run("unload", worker_prompt("unload"))
        time.sleep(3)
        wid = j1.get("worker_id")
        report["checks"]["unload"] = {
            "unloaded": un.get("worker_report", {}).get("unloaded"),
            "leftover": wsl_procs(distro, wid) if wid else None,
            "gpu_used_mib": base.gpu_used_mib(),
        }
        # 3) after unload a new worker starts; cancel during generation keeps (or replaces) it, next job recovers
        a1 = r.run("A1 after unload (new worker)", generate_prompt(rid, "monarch_h2", P1, 101), free=True)
        report["checks"]["new_worker_after_unload"] = a1.get("started_for_this_job") is True and a1.get("worker_pid") != j1.get("worker_pid")
        report["checks"]["after_unload_identical_output"] = a1["videos"][0]["sha256"] == j1["videos"][0]["sha256"]
        q = r.comfy.queue(generate_prompt(rid, "dense", P2, 2026, videos=4, prefix="persist_cancel"))
        time.sleep(12)
        base.http(r.comfy.base, "/interrupt", {})
        hc = r.comfy.wait(q, 900)
        report["checks"]["cancel"] = {"status": hc["status"]["status_str"], "messages": [m[0] for m in hc["status"].get("messages", [])]}
        a2 = r.run("A2 after cancel", generate_prompt(rid, "monarch_h2", P2, 101), free=True)
        report["checks"]["recovered_after_cancel"] = {
            "status": a2["status"],
            "same_worker": a2.get("worker_pid") == a1.get("worker_pid"),
            "identical_to_J2": a2["videos"][0]["sha256"] == j2["videos"][0]["sha256"] if a2.get("videos") else None,
        }
        # 4) worker crash while idle -> next job starts a new worker
        pid = a2.get("worker_pid")
        if pid and wsl_procs(distro, a2.get("worker_id")):
            subprocess.run(["wsl.exe", "-d", distro, "--exec", "/bin/kill", "-9", str(pid)], capture_output=True)
        time.sleep(2)
        a3 = r.run("A3 after worker crash (idle)", generate_prompt(rid, "monarch_h2", P3, 2026), free=True)
        report["checks"]["recovered_after_idle_crash"] = {"status": a3["status"], "new_worker": a3.get("worker_pid") not in (None, pid)}
        # 5) worker crash while busy -> that job fails, the next one works
        q = r.comfy.queue(generate_prompt(rid, "dense", P1, 55, videos=3, prefix="persist_crash"))
        time.sleep(15)
        live = wsl_procs(distro, a3.get("worker_id") or "none")
        if live:
            subprocess.run(["wsl.exe", "-d", distro, "--exec", "/bin/kill", "-9", str(a3.get("worker_pid"))], capture_output=True)
        hb = r.comfy.wait(q, 900)
        report["checks"]["busy_crash"] = {"status": hb["status"]["status_str"], "messages": [m[0] for m in hb["status"].get("messages", [])]}
        a4 = r.run("A4 after busy crash", generate_prompt(rid, "dense", P1, 101), free=True)
        report["checks"]["recovered_after_busy_crash"] = {
            "status": a4["status"],
            "identical_to_J4": a4["videos"][0]["sha256"] == j4["videos"][0]["sha256"] if a4.get("videos") else None,
        }
        # 6) runtime error (missing checkpoint) and timeout, then recovery on the main runtime
        e1 = r.run("E1 missing checkpoint", generate_prompt("missingckpt", "dense", P1, 1), expect="error")
        e2 = r.run("E2 1-minute timeout", generate_prompt("timeout1", "monarch_h1", P3, 9, videos=4), expect="error")
        a5 = r.run("A5 after errors", generate_prompt(rid, "dense", P2, 2026), free=True)
        report["checks"]["errors"] = {
            "missing_ckpt": [e1["status"], (e1["error"] or "")[:200]],
            "timeout": [e2["status"], (e2["error"] or "")[:200]],
            "recovered": a5["status"],
        }
        # 7) one-shot comparison inside ComfyUI (each job loads the model again)
        o1 = r.run("O1 one-shot monarch_h2 P1 s101", generate_prompt("oneshot", "monarch_h2", P1, 101), free=True)
        o2 = r.run("O2 one-shot monarch_h2 P2 s101", generate_prompt("oneshot", "monarch_h2", P2, 101))
        report["checks"]["one_shot_vs_persistent_same_output"] = o1["videos"][0]["sha256"] == j1["videos"][0]["sha256"] if o1.get("videos") else None
        report["checks"]["one_shot_loads_every_job"] = [o.get("model_load_seconds") for o in (o1, o2)]
        # 8) idle timeout: a worker with a 60 s idle limit exits by itself
        r.run("unload before idle test", worker_prompt("unload"))
        i1 = r.run("I1 idle60 job", generate_prompt("idle60", "dense", P1, 101), free=True)
        time.sleep(80)
        report["checks"]["idle_timeout"] = {"leftover": wsl_procs(distro, i1.get("worker_id") or "none"), "gpu_used_mib": base.gpu_used_mib()}
        # 9) ComfyUI crash while the worker is idle, then while it is busy
        i2 = r.run("C1 worker for idle-crash test", generate_prompt(rid, "dense", P1, 101), free=True)
        r.kill_comfy_hard()
        time.sleep(10)
        report["checks"]["comfy_crash_idle"] = {"leftover": wsl_procs(distro, i2.get("worker_id") or "none"), "gpu_used_mib": base.gpu_used_mib()}
        r.start_comfy()
        i3 = r.run("C2 worker for busy-crash test", generate_prompt(rid, "dense", P1, 101))
        r.comfy.queue(generate_prompt(rid, "dense", P3, 77, videos=4, prefix="persist_busycrash"))
        time.sleep(20)
        r.kill_comfy_hard()
        time.sleep(10)
        report["checks"]["comfy_crash_busy"] = {"leftover": wsl_procs(distro, i3.get("worker_id") or "none"), "gpu_used_mib": base.gpu_used_mib()}
    finally:
        if r.comfy is not None and r.comfy.proc.poll() is None:
            try:
                r.run("final unload", worker_prompt("unload"))
            except Exception:
                pass
            r.comfy.stop()
        report["records"] = r.records
        report["gpu_idle_mib_after"] = base.gpu_used_mib()
        args.out.write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report["checks"], indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
