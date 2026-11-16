"""Generate comparison chart for old imbalanced dataset vs new 3k balanced dataset."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    df = pd.read_csv(root / "artifacts" / "komparasi_model_3k.csv")
    with open(root / "artifacts" / "ensemble_metadata_3k.json", encoding="utf-8") as f:
        meta = json.load(f)

    ensemble_auc = meta["test_metrics"]["roc_auc"]

    # Urutkan berdasarkan AUC (Baru) menurun agar visual lebih rapi
    df = df.sort_values("AUC (Baru)", ascending=True).reset_index(drop=True)

    models = df["Model"].tolist()
    y = np.arange(len(models))
    h = 0.36

    plt.rcParams.update({
        "font.size": 9.5,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.labelsize": 9.5,
    })

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 5.2), dpi=300)
    fig.patch.set_facecolor("white")

    # Panel A: Recall (Lama vs Baru) pada ambang standar t = 0.50
    b1 = ax1.barh(y - h / 2, df["Recall (Lama)"] * 100, height=h, color="#94a3b8", label="Dataset Lama (4,87% stroke)")
    b2 = ax1.barh(y + h / 2, df["Recall (Baru)"] * 100, height=h, color="#2563eb", label="Dataset 3k Seimbang (26,10% stroke)")

    for bar in b1:
        val = bar.get_width()
        ax1.text(val + 1.0, bar.get_y() + bar.get_height() / 2, f"{val:.1f}%", va="center", fontsize=8, color="#475569")
    for bar in b2:
        val = bar.get_width()
        ax1.text(val + 1.0, bar.get_y() + bar.get_height() / 2, f"{val:.1f}%", va="center", fontsize=8, fontweight="bold", color="#1e3a8a")

    ax1.set_yticks(y)
    ax1.set_yticklabels(models)
    ax1.set_xlim(0, 68)
    ax1.set_xlabel("Recall / Sensitivitas (%) pada Ambang t = 0,50")
    ax1.set_title("A. Pembebasan dari Zero-Recall Trap")
    ax1.grid(axis="x", linestyle="--", alpha=0.4)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=1, frameon=True, fontsize=8.5)

    # Panel B: ROC-AUC (Lama vs Baru) + Garis Ensemble Soft Voting 3k
    b3 = ax2.barh(y - h / 2, df["AUC (Lama)"], height=h, color="#94a3b8", label="Dataset Lama (AUC)")
    b4 = ax2.barh(y + h / 2, df["AUC (Baru)"], height=h, color="#0d9488", label="Dataset 3k Seimbang (AUC)")

    for bar in b3:
        val = bar.get_width()
        ax2.text(val + 0.006, bar.get_y() + bar.get_height() / 2, f"{val:.3f}", va="center", fontsize=8, color="#475569")
    for bar in b4:
        val = bar.get_width()
        ax2.text(val + 0.006, bar.get_y() + bar.get_height() / 2, f"{val:.3f}", va="center", fontsize=8, fontweight="bold", color="#115e59")

    ax2.axvline(ensemble_auc, color="#dc2626", linestyle="--", linewidth=1.6, label=f"Ensemble Soft Voting 3k ({ensemble_auc:.3f})")
    ax2.set_yticks(y)
    ax2.set_yticklabels([])
    ax2.set_xlim(0.50, 0.96)
    ax2.set_xlabel("Skor ROC-AUC")
    ax2.set_title("B. Peningkatan Daya Pisah (ROC-AUC)")
    ax2.grid(axis="x", linestyle="--", alpha=0.4)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=1, frameon=True, fontsize=8.5)

    fig.suptitle(
        "Perbandingan Kinerja 6 Algoritma: Dataset Lama vs Dataset Baru 3k (Fitur Dasar)",
        fontsize=12.5,
        fontweight="bold",
        y=0.97,
    )
    plt.tight_layout(rect=[0, 0.04, 1, 0.93])

    out_path = root / "assets" / "evaluasi_14" / "fig_komparasi_lama_vs_3k.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
