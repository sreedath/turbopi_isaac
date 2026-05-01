"""Plot train/val loss curves from training_summary.json."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", required=True, help="Path to training_summary.json")
    parser.add_argument("--out-dir", default=None, help="Where to save the loss plots (defaults to summary's parent).")
    args = parser.parse_args()

    summary_path = Path(args.summary)
    payload = json.loads(summary_path.read_text())
    history = payload.get("history", [])
    if not history:
        print("[plot] No history entries; nothing to plot.")
        return

    epochs = [r["epoch"] for r in history]
    train_loss = [r["train_loss"] for r in history]
    val_loss = [r["val_loss"] for r in history]
    val_loss_clean = [v if not (v is None or (isinstance(v, float) and math.isnan(v))) else float("nan") for v in val_loss]

    out_dir = Path(args.out_dir) if args.out_dir else summary_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(7, 4))
    plt.plot(epochs, train_loss, label="train", marker="o")
    if any(not math.isnan(v) for v in val_loss_clean):
        plt.plot(epochs, val_loss_clean, label="val", marker="s")
    plt.xlabel("epoch")
    plt.ylabel("Huber loss")
    plt.title("Training/Validation loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plot_path = out_dir / "loss_curve.png"
    plt.savefig(plot_path, dpi=120)
    print(f"[plot] saved {plot_path}")

    if any(r.get("val_mae_vx") is not None for r in history):
        plt.figure(figsize=(7, 4))
        for key, color in (("val_mae_vx", "C0"), ("val_mae_vy", "C1"), ("val_mae_omega", "C2")):
            ys = [r.get(key, float("nan")) for r in history]
            plt.plot(epochs, ys, label=key.replace("val_", ""), color=color, marker=".")
        plt.xlabel("epoch")
        plt.ylabel("|pred - target| (normalized)")
        plt.title("Validation MAE per axis")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        mae_path = out_dir / "val_mae_curve.png"
        plt.savefig(mae_path, dpi=120)
        print(f"[plot] saved {mae_path}")


if __name__ == "__main__":
    main()
