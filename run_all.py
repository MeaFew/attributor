"""Windows-compatible one-shot pipeline runner.

Replaces `make all` on systems without GNU Make (e.g., Windows).
Usage: python run_all.py [--output PREFIX]
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

# Force UTF-8 mode for child processes on Windows before any heavy imports.
os.environ.setdefault("PYTHONUTF8", "1")


def run(cmd: list[str], cwd: Path | None = None):
    print(f"\n{'=' * 60}")
    print(f">>> {' '.join(cmd)}")
    print("=" * 60)
    # cmd is a list; no shell=True — avoids shell-injection surface and
    # correctly handles paths with spaces.
    result = subprocess.run(cmd, cwd=cwd)
    if result.returncode != 0:
        print(f"WARNING: Command failed with exit code {result.returncode}")
        return False
    return True


def main():
    parser = argparse.ArgumentParser(description="Marketing Attribution full pipeline runner")
    parser.add_argument("--output", type=str, default=None, help="Output path prefix for generated files")
    args = parser.parse_args()

    here = Path(__file__).resolve().parent

    # Build step commands as argv lists; append --output if provided.
    preprocess_cmd = ["python", "scripts/preprocess.py"]
    if args.output:
        preprocess_cmd += ["--output", args.output]
    steps = [
        ("Preprocessing", preprocess_cmd),
        ("MMM Modeling", ["python", "scripts/mmm_model.py"]),
        ("Touchpoint Generation", ["python", "scripts/generate_touchpoints.py"]),
        ("Multi-touch Attribution", ["python", "scripts/multi_touch_attribution.py"]),
        ("Budget Optimization", ["python", "scripts/budget_optimizer.py"]),
    ]

    print("Marketing Attribution & Budget Optimization - Full Pipeline")
    print("=" * 60)

    for name, cmd in steps:
        if not run(cmd, cwd=here):
            print(f"\nPipeline stopped at step: {name}")
            sys.exit(1)

    print("\n" + "=" * 60)
    print("Pipeline completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
