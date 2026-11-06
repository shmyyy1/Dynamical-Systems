"""Run the lab scripts once each and save their logs and result summary."""

import argparse
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import time
from datetime import datetime, timezone


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parts", nargs="+", type=int, choices=[2, 3, 4, 5], default=[2, 3, 4, 5])
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/latest"))
    args = parser.parse_args()
    if args.epochs < 1 or args.threads < 1:
        parser.error("--epochs and --threads must be positive")

    root = Path(__file__).resolve().parent
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / "summary.json").exists() or any(output.glob("part*.log")):
        parser.error("output directory already contains a run; choose another --output-dir")
    env = os.environ.copy()
    env.update({
        "GRAPHCONTROL_OUTPUT_DIR": str(output),
        "GRAPHCONTROL_EPOCHS": str(args.epochs),
        "GRAPHCONTROL_SEED": str(args.seed),
        "GRAPHCONTROL_THREADS": str(args.threads),
        "PYTHONUNBUFFERED": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "OMP_NUM_THREADS": str(args.threads),
        "OPENBLAS_NUM_THREADS": str(args.threads),
        "MKL_NUM_THREADS": str(args.threads),
    })
    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, text=True,
                              capture_output=True, check=True).stdout.strip()
    summary = {
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_revision": revision,
        "source_note": "Local portability and plotting fixes; see git diff.",
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {name: importlib.metadata.version(name)
                     for name in ["numpy", "scipy", "matplotlib", "torch"]},
        "epochs": args.epochs,
        "seed_per_part": args.seed,
        "threads": args.threads,
        "parts": {},
    }

    def save_summary():
        (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    for part in dict.fromkeys(args.parts):
        print(f"Running Part{part}.py ({args.epochs} epochs for GNNs)...", flush=True)
        start = time.perf_counter()
        log_path = output / f"part{part}.log"
        with log_path.open("w", encoding="utf-8") as log:
            result = subprocess.run([sys.executable, "-u", f"Part{part}.py"], cwd=root,
                                    env=env, stdout=log, stderr=subprocess.STDOUT)
        text = log_path.read_text(encoding="utf-8")
        costs = {label: float(value) for label, value in re.findall(
            r"Cost for ([^:\n]+):\s*([-+0-9.eE]+)", text)}
        expected_costs = 2 if part == 2 else 1
        success = result.returncode == 0 and len(costs) == expected_costs and all(
            math.isfinite(value) for value in costs.values())
        summary["parts"][str(part)] = {
            "exit_code": result.returncode,
            "success": success,
            "elapsed_seconds": round(time.perf_counter() - start, 3),
            "costs": costs,
            "figures": [str(path.relative_to(output).as_posix())
                        for path in sorted((output / f"part{part}").rglob("*.png"))],
            "checkpoints": [str(path.relative_to(output).as_posix())
                            for path in sorted((output / f"part{part}").rglob("*.ckpt"))],
        }
        save_summary()
        if not success:
            print(text[-6000:])
            raise SystemExit(f"Part {part} failed; see {log_path}")
        print(f"Part {part}: {costs}; {summary['parts'][str(part)]['elapsed_seconds']:.1f}s", flush=True)
    summary["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    save_summary()
    print(f"Saved logs, figures, checkpoints, and summary to {output}")


if __name__ == "__main__":
    main()
