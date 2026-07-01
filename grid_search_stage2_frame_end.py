"""
Grid-search memory gate initialization for train_stage2_frame_end.py.

Each trial runs train_stage2_frame_end.py with memory enabled. The child
process writes directly to the terminal, so tqdm progress bars stay normal
instead of being reprinted through a pipe.

A compact CSV summary is written from metrics emitted by the training script:

    checkpoints_s2_frame_end_grid/grid_metrics.csv
    checkpoints_s2_frame_end_grid/grid_summary.csv

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
import subprocess
import sys
from pathlib import Path


CONTROLLED_OPTIONS_WITH_VALUES = {
    "--epochs",
    "--save_root",
    "--memory_gate_init",
    "--knn_k",
    "--metrics_csv",
    "--run_name",
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


def run_trial(
    python_exe: str,
    train_script: Path,
    run_name: str,
    gate_init: float,
    knn_k: int,
    epochs: int,
    save_root: Path,
    metrics_csv: Path,
    passthrough_args: list[str],
    dry_run: bool,
) -> int:
    trial_save_root = save_root / run_name
    cmd = [
        python_exe,
        str(train_script),
        "--use_memory_bank",
        "--memory_gate_init", str(gate_init),
        "--knn_k", str(knn_k),
        "--epochs", str(epochs),
        "--save_root", str(trial_save_root),
        "--metrics_csv", str(metrics_csv),
        "--run_name", run_name,
        *passthrough_args,
    ]

    print("\n" + "=" * 80)
    print(f"FRAME-END GRID TRIAL: {run_name}")
    print(" ".join(cmd))
    print("=" * 80)

    if dry_run:
        return 0

    trial_save_root.mkdir(parents=True, exist_ok=True)
    return subprocess.run(cmd).returncode


def write_summary(metrics_csv: Path, summary_csv: Path):
    if not metrics_csv.exists():
        print(f"No metrics CSV found at {metrics_csv}; skipping summary.")
        return

    with open(metrics_csv, newline="") as f:
        rows = list(csv.DictReader(f))

    by_run_epoch: dict[str, dict[int, dict[str, dict]]] = {}
    for row in rows:
        run_name = row["run_name"]
        epoch = int(row["epoch"])
        key = f"{row['split']}_{row['mode']}"
        by_run_epoch.setdefault(run_name, {}).setdefault(epoch, {})[key] = row

    summary_rows = []
    for run_name, epochs in by_run_epoch.items():
        best_epoch = None
        best_auc = -1.0
        for epoch, metrics in epochs.items():
            row = metrics.get("val_frame_end_w_mem")
            if row is None:
                continue
            auc = float(row["auc"])
            if auc > best_auc:
                best_auc = auc
                best_epoch = epoch

        if best_epoch is None:
            continue

        metrics = epochs[best_epoch]

        def auc(key: str) -> str:
            row = metrics.get(key)
            return row["auc"] if row else ""

        val_ref = metrics.get("val_frame_end_w_mem", {})
        summary_rows.append({
            "run_name": run_name,
            "gate_init": val_ref.get("gate_init", ""),
            "knn_k": val_ref.get("knn_k", ""),
            "best_val_epoch": best_epoch,
            "best_val_frame_end_w_mem_auc": auc("val_frame_end_w_mem"),
            "best_val_frame_end_w_o_mem_auc": auc("val_frame_end_w_o_mem"),
            "best_val_no_frame_w_mem_auc": auc("val_no_frame_w_mem"),
            "best_val_no_frame_w_o_mem_auc": auc("val_no_frame_w_o_mem"),
            "test_frame_end_w_mem_auc": auc("test_frame_end_w_mem"),
            "test_frame_end_w_o_mem_auc": auc("test_frame_end_w_o_mem"),
            "test_no_frame_w_mem_auc": auc("test_no_frame_w_mem"),
            "test_no_frame_w_o_mem_auc": auc("test_no_frame_w_o_mem"),
            "gate_avg": val_ref.get("gate_avg", ""),
        })

    if not summary_rows:
        print(f"No completed metric rows found in {metrics_csv}.")
        return

    summary_rows.sort(
        key=lambda row: float(row["best_val_frame_end_w_mem_auc"] or "-1"),
        reverse=True,
    )
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    best = summary_rows[0]
    print(
        f"\nBest by validation frame-end AUC: {best['run_name']} "
        f"at epoch {best['best_val_epoch']} "
        f"(w/ mem AUC={best['best_val_frame_end_w_mem_auc']}, "
        f"w/o mem AUC={best['best_val_frame_end_w_o_mem_auc']})"
    )
    print(f"Wrote summary: {summary_csv}")


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
    metrics_csv = save_root / "grid_metrics.csv"
    summary_csv = save_root / "grid_summary.csv"
    train_script = Path(args.train_script)

    failures = []
    for gate_init in parse_float_list(args.gate_inits):
        run_name = f"frame_end_mem_gate_{format_value(gate_init)}"
        code = run_trial(
            args.python,
            train_script,
            run_name,
            gate_init,
            args.knn_k,
            args.epochs_per_trial,
            save_root,
            metrics_csv,
            passthrough,
            args.dry_run,
        )
        if code != 0:
            failures.append((run_name, code))

    if not args.dry_run:
        write_summary(metrics_csv, summary_csv)

    if failures:
        print("\nFailed trials:")
        for run_name, code in failures:
            print(f"  {run_name}: exit code {code}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
