import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

OUT_DIR = Path("Rainfall_Prediction/output")

def main():
    processed = OUT_DIR / "processed.csv"
    if not processed.exists():
        print("[ERR] Run preprocessing.py first.")
        return
    df = pd.read_csv(processed)

    corr = df.corr(numeric_only=True)
    mask = (corr.where(~corr.isna(), 0)).mask(~(corr.abs() > 0), other=0)  # keep all, just clean NaNs

    plt.figure(figsize=(16,12))
    sns.heatmap(corr, cmap="coolwarm", center=0, linewidths=0.2)
    plt.title("Correlation Heatmap (Processed)")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "correlation_heatmap.png", dpi=150)
    plt.close()
    print(f"[OK] Saved {OUT_DIR/'correlation_heatmap.png'}")

if __name__ == "__main__":
    main()
