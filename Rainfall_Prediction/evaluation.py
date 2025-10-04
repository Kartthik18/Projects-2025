import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

OUT_DIR = Path("Rainfall_Prediction/output")

def main():
    metrics_path = OUT_DIR / "model_metrics.csv"
    if not metrics_path.exists():
        print("[ERR] Run train_models.py first.")
        return

    df = pd.read_csv(metrics_path)

    # Accuracy vs Time
    fig, ax1 = plt.subplots(figsize=(10,6))
    ax2 = ax1.twinx()

    sns.barplot(x="Model", y="Time_sec", data=df, ax=ax1, palette="Blues")
    sns.lineplot(x="Model", y="Accuracy", data=df, ax=ax2, color="red", marker="o")

    ax1.set_ylabel("Time (sec)")
    ax2.set_ylabel("Accuracy")
    ax1.set_title("Model Comparison: Accuracy vs Time")
    plt.xticks(rotation=20)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "model_accuracy_time.png", dpi=150)
    plt.close(fig)

    # ROC_AUC vs Kappa
    fig, ax1 = plt.subplots(figsize=(10,6))
    ax2 = ax1.twinx()

    sns.barplot(x="Model", y="ROC_AUC", data=df, ax=ax1, palette="Greens")
    sns.lineplot(x="Model", y="Cohen_Kappa", data=df, ax=ax2, color="purple", marker="o")

    ax1.set_ylabel("ROC_AUC")
    ax2.set_ylabel("Cohen Kappa")
    ax1.set_title("Model Comparison: ROC_AUC vs Cohen Kappa")
    plt.xticks(rotation=20)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "roc_vs_kappa.png", dpi=150)
    plt.close(fig)

    print(f"[OK] Saved plots to {OUT_DIR}")

if __name__ == "__main__":
    main()
