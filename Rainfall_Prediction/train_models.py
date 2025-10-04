import json
import time
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.metrics import accuracy_score, roc_auc_score, cohen_kappa_score, classification_report, roc_curve, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier

import xgboost as xgb
import lightgbm as lgb
import catboost as cb
import matplotlib.pyplot as plt
import seaborn as sns

OUT_DIR = Path("Rainfall_Prediction/output")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def plot_roc_cur(fper, tper, title, outpath):
    plt.figure()
    plt.plot(fper, tper, label="ROC", lw=2)
    plt.plot([0, 1], [0, 1], linestyle="--", color="grey")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=140)
    plt.close()

def run_model(model, X_train, y_train, X_test, y_test, name: str, verbose: bool=True):
    t0 = time.time()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "ROC_AUC": roc_auc_score(y_test, y_pred),
        "Cohen_Kappa": cohen_kappa_score(y_test, y_pred),
        "Time_sec": time.time() - t0
    }
    print(f"\n== {name} ==")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
    print(classification_report(y_test, y_pred, digits=4))

    # Probs for ROC
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_test)[:, 1]
    else:
        # Some models may not have predict_proba under certain configs
        probs = y_pred
    fper, tper, _ = roc_curve(y_test, probs)
    plot_roc_cur(fper, tper, f"ROC - {name}", OUT_DIR / f"roc_{name}.png")

    # Confusion matrix
    fig, ax = plt.subplots(figsize=(4.5,4))
    ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, cmap=plt.cm.Blues, normalize="all", ax=ax)
    ax.set_title(f"Confusion Matrix - {name}")
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"cm_{name}.png", dpi=140)
    plt.close(fig)

    return metrics

def main():
    X_train = pd.read_csv(OUT_DIR / "X_train.csv")
    X_test  = pd.read_csv(OUT_DIR / "X_test.csv")
    y_train = pd.read_csv(OUT_DIR / "y_train.csv").squeeze()
    y_test  = pd.read_csv(OUT_DIR / "y_test.csv").squeeze()

    # Use feature selection union if available
    selected_path = OUT_DIR / "selected_features.json"
    if selected_path.exists():
        with open(selected_path) as f:
            features = json.load(f)["union_selected"]
        if features:
            X_train = X_train[features]
            X_test  = X_test[features]

    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    models = {
        "LogisticRegression": LogisticRegression(penalty="l1", solver="liblinear", max_iter=200),
        "DecisionTree": DecisionTreeClassifier(max_depth=16, max_features="sqrt", random_state=12345),
        "NeuralNet": MLPClassifier(hidden_layer_sizes=(30,30,30), activation="logistic", solver="lbfgs", max_iter=500, random_state=12345),
        "RandomForest": RandomForestClassifier(n_estimators=100, max_depth=16, random_state=12345, n_jobs=-1),
        "LightGBM": lgb.LGBMClassifier(colsample_bytree=0.95, max_depth=16, min_split_gain=0.1, n_estimators=200,
                                       num_leaves=50, reg_alpha=1.2, reg_lambda=1.2, subsample=0.95, subsample_freq=20, random_state=12345, verbose=-1),
        "CatBoost": cb.CatBoostClassifier(iterations=50, depth=8, verbose=0, random_state=12345),
        "XGBoost": xgb.XGBClassifier(n_estimators=500, max_depth=16, eval_metric="logloss", n_jobs=-1, random_state=12345)
    }

    rows = []
    for name, model in models.items():
        m = run_model(model, X_train, y_train, X_test, y_test, name)
        m["Model"] = name
        rows.append(m)

    df = pd.DataFrame(rows).sort_values(by="Accuracy", ascending=False)
    df.to_csv(OUT_DIR / "model_metrics.csv", index=False)
    print(f"\n[OK] Saved metrics to {OUT_DIR/'model_metrics.csv'}")

if __name__ == "__main__":
    main()
