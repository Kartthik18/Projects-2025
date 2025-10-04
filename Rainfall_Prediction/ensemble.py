import json
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
import catboost as cb
import xgboost as xgb

from mlxtend.classifier import EnsembleVoteClassifier
from mlxtend.plotting import plot_decision_regions
import matplotlib.pyplot as plt
import itertools
import matplotlib.gridspec as gridspec

OUT_DIR = Path("Rainfall_Prediction/output")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def main():
    X_train = pd.read_csv(OUT_DIR / "X_train.csv")
    X_test  = pd.read_csv(OUT_DIR / "X_test.csv")
    y_train = pd.read_csv(OUT_DIR / "y_train.csv").squeeze()
    y_test  = pd.read_csv(OUT_DIR / "y_test.csv").squeeze()

    # If feature selection union exists, use it
    sel_path = OUT_DIR / "selected_features.json"
    if sel_path.exists():
        with open(sel_path) as f:
            feats = json.load(f)["union_selected"]
        if feats:
            X_train = X_train[feats]
            X_test = X_test[feats]

    scaler = StandardScaler()
    X = scaler.fit_transform(pd.concat([X_train, X_test], axis=0))
    y = pd.concat([y_train, y_test], axis=0).to_numpy()

    # Base learners (tree-based to avoid probability issues)
    clf_rf  = RandomForestClassifier(n_estimators=200, random_state=12345, n_jobs=-1)
    clf_lgb = lgb.LGBMClassifier(n_estimators=200, random_state=12345, verbose=-1)
    clf_cb  = cb.CatBoostClassifier(iterations=100, depth=8, random_state=12345, verbose=0)
    clf_xgb = xgb.XGBClassifier(n_estimators=300, eval_metric="logloss", random_state=12345, n_jobs=-1)

    eclf = EnsembleVoteClassifier(clfs=[clf_rf, clf_lgb, clf_cb, clf_xgb],
                                  weights=[1,1,1,1],
                                  voting='soft')

    # For decision region visualization we need 2 or 3 dims; pick 3 most informative if available
    cols = list(range(X.shape[1]))
    if X.shape[1] >= 3:
        sel_dims = cols[:3]
    elif X.shape[1] == 2:
        sel_dims = cols
    else:
        print("[INFO] Need >=2 features to draw decision regions; skipping plot.")
        return

    X_vis = X[:, sel_dims]
    y_vis = y.astype(np.int32)

    fig = plt.figure(figsize=(8, 6))
    plot_decision_regions(X=X_vis, y=y_vis, clf=eclf,
                          filler_feature_values={},
                          filler_feature_ranges={},
                          legend=2)
    plt.title("Ensemble (RF+LGBM+CatBoost+XGB) Decision Regions")
    plt.tight_layout()
    fig.savefig(OUT_DIR / "ensemble_decision_regions.png", dpi=150)
    plt.close(fig)
    print(f"[OK] Saved ensemble decision region plot to {OUT_DIR/'ensemble_decision_regions.png'}")

if __name__ == "__main__":
    main()
