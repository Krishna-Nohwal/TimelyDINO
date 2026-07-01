"""
Grid-search memory gate initialization for train_stage2_frame_end.py.

Each trial runs train_stage2_frame_end.py with memory enabled, streams the
training logs to the console, and also saves them under:

    checkpoints_s2_frame_end_grid/logs/

By default this is intentionally small: four gate values, k=32, and 10 epochs
per version. The first gate value is 0.13.

Example
-------
python grid_search_stage2_frame_end.py ^
    --gate_inits 0.13,0.20,0.35,0.50 ^
    --epochs_per_trial 10 ^
    -- --load_from checkpoints_vit_4layers/best.pth ^
       --manifest E:/Work/sampled_30k/manifest_onct.csv ^
       --root_dir E:/Work/sampled_30k/ ^
       --cdf_root E:/Work/cdfv1_onct_out ^
       --cdf_csv E:/Work/cdfv1_onct_out/manifest_cdfv1_onct.csv ^
       --num_frames 32 --batch_size 10 --num_workers 20 --no_compile
"""

import argparse
import csv
import re
import subprocess
import sys
from pathlib import Path


CONTROLLED_OPTIONS_WITH_VALUES = {
    "--epochs",
    "--save_root",
    "--memory_gate_init",
    "--knn_k",
}
CONTROLLED_FLAGS = {"--use_memory_bank"}


def parse_float_list(text: str) -> list[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def strip_controlled_args(args: list[str]) -> list[str]:
    cleaned = []
    i = 0
    while i < len(args):
        arg = args[i]
        if arg in CONTROLLED_FLAGS:
            i += 1
            continue
        if arg in CONTROLLED_OPTIONS_WITH_VALUES:
            i += 2
            continue
        if any(arg.startswith(opt + "=") for opt in CONTROLLED_OPTIONS_WITH_VALUES):
            i += 1
            continue
        cleaned.append(arg)
        i += 1
    return cleaned


def format_value(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def parse_log_summary(log_path: Path) -> dict:
    summary = {
        "best_val_frame_end_auc": "",
        "best_val_no_frame_auc": "",
        "best_val_epoch": "",
        "last_test_frame_end_auc": "",
        "last_test_no_frame_auc": "",
        "last_gate_avg": "",
    }
    val_re = re.compile(
        r"\[Val AUC summary\]\s+frame in end=([0-9.]+)\s+no frame in end=([0-9.]+)"
    )
    test_re = re.compile(
        r"\[Test AUC summary\]\s+frame in end=([0-9.]+)\s+no frame in end=([0-9.]+)"
    )
    epoch_re = re.compile(r"EPOCH\s+(\d+)/")
    gate_re = re.compile(r"Memory gate:\s+avg=([0-9.]+)")

    current_epoch = ""
    best_val = -1.0

    with open(log_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            epoch_match = epoch_re.search(line)
            if epoch_match:
                current_epoch = epoch_match.group(1)

            gate_match = gate_re.search(line)
            if gate_match:
                summary["last_gate_avg"] = gate_match.group(1)

            val_match = val_re.search(line)
            if val_match:
                val_frame = float(val_match.group(1))
                if val_frame > best_val:
                    best_val = val_frame
                    summary["best_val_frame_end_auc"] = val_match.group(1)
                    summary["best_val_no_frame_auc"] = val_match.group(2)
                    summary["best_val_epoch"] = current_epoch

            test_match = test_re.search(line)
            if test_match:
                summary["last_test_frame_end_auc"] = test_match.group(1)
                summary["last_test_no_frame_auc"] = test_match.group(2)

    return summary


def run_trial(
    python_exe: str,
    train_script: Path,
    run_name: str,
    gate_init: float,
    knn_k: int,
    epochs: int,
    save_root: Path,
    log_dir: Path,
    passthrough_args: list[str],
    dry_run: bool,
) -> tuple[int, Path]:
    trial_save_root = save_root / run_name
    log_path = log_dir / f"{run_name}.log"

    cmd = [
        python_exe,
        str(train_script),
        "--use_memory_bank",
        "--memory_gate_init", str(gate_init),
        "--knn_k", str(knn_k),
        "--epochs", str(epochs),
        "--save_root", str(trial_save_root),
        *passthrough_args,
    ]

    print("\n" + "=" * 80)
    print(f"FRAME-END GRID TRIAL: {run_name}")
    print(" ".join(cmd))
    print("=" * 80)

    if dry_run:
        return 0, log_path

    trial_save_root.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    with open(log_path, "w", encoding="utf-8", errors="replace") as log_file:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="")
            log_file.write(line)
        return proc.wait(), log_path


def main():
    parser = argparse.ArgumentParser(
        description="Grid search train_stage2_frame_end.py memory gate initialization."
    )
    parser.add_argument("--train_script", default="train_stage2_frame_end.py", type=str)
    parser.add_argument("--python", default=sys.executable, type=str)
    parser.add_argument("--save_root", default="checkpoints_s2_frame_end_grid", type=str)
    parser.add_argument("--epochs_per_trial", default=10, type=int)
    parser.add_argument("--gate_inits", default="0.13,0.20,0.35,0.50", type=str)
    parser.add_argument("--knn_k", default=32, type=int)
    parser.add_argument("--dry_run", action="store_true")
    args, passthrough = parser.parse_known_args()

    if passthrough and passthrough[0] == "--":
        passthrough = passthrough[1:]
    passthrough = strip_controlled_args(passthrough)

    save_root = Path(args.save_root)
    log_dir = save_root / "logs"
    summary_csv = save_root / "grid_summary.csv"
    train_script = Path(args.train_script)

    rows = []
    failures = []
    for gate_init in parse_float_list(args.gate_inits):
        run_name = f"frame_end_mem_gate_{format_value(gate_init)}"
        code, log_path = run_trial(
            args.python,
            train_script,
            run_name,
            gate_init,
            args.knn_k,
            args.epochs_per_trial,
            save_root,
            log_dir,
            passthrough,
            args.dry_run,
        )
        if code != 0:
            failures.append((run_name, code))
            continue
        if not args.dry_run:
            rows.append({
                "run_name": run_name,
                "gate_init": gate_init,
                "knn_k": args.knn_k,
                "epochs": args.epochs_per_trial,
                "log_path": str(log_path),
                **parse_log_summary(log_path),
            })

    if rows:
        save_root.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            "run_name",
            "gate_init",
            "knn_k",
            "epochs",
            "best_val_epoch",
            "best_val_frame_end_auc",
            "best_val_no_frame_auc",
            "last_test_frame_end_auc",
            "last_test_no_frame_auc",
            "last_gate_avg",
            "log_path",
        ]
        rows.sort(
            key=lambda row: float(row["best_val_frame_end_auc"] or "-1"),
            reverse=True,
        )
        with open(summary_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        best = rows[0]
        print(
            f"\nBest by validation frame-end AUC: {best['run_name']} "
            f"at epoch {best['best_val_epoch']} "
            f"(AUC={best['best_val_frame_end_auc']})"
        )
        print(f"Wrote summary: {summary_csv}")

    if failures:
        print("\nFailed trials:")
        for run_name, code in failures:
            print(f"  {run_name}: exit code {code}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
