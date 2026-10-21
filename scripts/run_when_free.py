"""Wait until the GPU has enough free memory, then launch the sweeps.

Lets you queue the benchmark tonight and leave. It polls nvidia-smi and only
starts once the card has been quiet for a sustained period -- a single free
reading is not enough, because a job between epochs can briefly look idle.

    python scripts\\run_when_free.py --need-gb 14

Stop it any time with Ctrl-C; nothing is running yet while it waits.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def free_gb() -> float | None:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            text=True, stderr=subprocess.DEVNULL)
        return int(out.strip().splitlines()[0]) / 1024.0
    except Exception:
        return None


def stamp() -> str:
    return datetime.now().strftime("%H:%M:%S")


def wait_for_gpu(need_gb: float, stable_checks: int, interval: int):
    print(f"[{stamp()}] waiting for >= {need_gb:.0f} GB free, "
          f"stable across {stable_checks} checks {interval}s apart")
    streak = 0
    while True:
        f = free_gb()
        if f is None:
            print(f"[{stamp()}] nvidia-smi unavailable -- proceeding anyway")
            return
        if f >= need_gb:
            streak += 1
            print(f"[{stamp()}] {f:.1f} GB free  ({streak}/{stable_checks})")
            if streak >= stable_checks:
                print(f"[{stamp()}] GPU is free. Starting.\n")
                return
        else:
            if streak:
                print(f"[{stamp()}] {f:.1f} GB free -- busy again, resetting")
            streak = 0
        time.sleep(interval)


def run(script: str, extra: list[str]):
    cmd = [sys.executable, str(ROOT / "scripts" / script), *extra]
    print(f"\n[{stamp()}] >>> {' '.join(cmd)}\n", flush=True)
    t0 = time.time()
    rc = subprocess.call(cmd, cwd=ROOT)
    mins = (time.time() - t0) / 60
    print(f"\n[{stamp()}] <<< {script} exited {rc} after {mins:.1f} min", flush=True)
    if rc != 0:
        print("Non-zero exit -- stopping the queue so you can inspect the error.")
        sys.exit(rc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--need-gb", type=float, default=14.0,
                    help="free VRAM required before starting")
    ap.add_argument("--interval", type=int, default=60, help="poll seconds")
    ap.add_argument("--stable-checks", type=int, default=5,
                    help="consecutive free readings required")
    ap.add_argument("--skip-wait", action="store_true")
    args = ap.parse_args()

    if not args.skip_wait:
        wait_for_gpu(args.need_gb, args.stable_checks, args.interval)

    # Pix2Pix generators first -- Experiment A's pix2pix arm depends on them.
    run("02_train_pix2pix.py", [])
    run("03_run_experiment_a.py", [])
    run("04_run_experiment_b.py", ["--arm", "classical", "--budget", "500", "--seed", "0"])
    run("05_make_figures.py", [])
    print(f"\n[{stamp()}] Queue complete. Figures are in the run folder.")


if __name__ == "__main__":
    main()
