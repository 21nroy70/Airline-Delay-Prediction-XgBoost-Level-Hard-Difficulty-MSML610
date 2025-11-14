"""
Train LightGBM and CatBoost baselines with time-aware split and robust OHE handling.

Artifacts in --out_dir:
- lightgbm_model.joblib, catboost_model.joblib
- baseline_metrics.json
- lgb_roc.png, lgb_pr.png
- cat_roc.png, cat_pr.png
"""

import os, json, argparse, warnings
from typing import List

import numpy as np
import pandas as pd
import joblib
import pyarrow.parquet as pq
import matplotlib.pyplot as plt

from sklearn.metrics import (
    classification_report, roc_auc_score, average_precision_score,
    precision_recall_curve, roc_curve
)
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

warnings.filterwarnings("ignore")
RANDOM_STATE = 610

CATEGORICAL: List[str] = ["AIRLINE","ORIGIN_AIRPORT","DESTINATION_AIRPORT","ORIGIN_STATE","DEST_STATE"]
NUMERIC: List[str] = ["DEPARTURE_DELAY","AIR_TIME","DISTANCE","ORIGIN_LAT","ORIGIN_LON","DEST_LAT","DEST_LON",
                      "temp","rhum","prcp","snow","wspd","pres"]

def read_parquet(path: str) -> pd.DataFrame:
    return pq.read_table(path).to_pandas()

def coerce(df: pd.DataFrame) -> pd.DataFrame:
    for c in NUMERIC + ["is_delayed"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df["is_delayed"] = df["is_delayed"].fillna(0).astype(int)
    if "FL_DATE" in df.columns:
        df["FL_DATE"] = pd.to_datetime(df["FL_DATE"], errors="coerce")
    return df

def time_split(df: pd.DataFrame, eval_size: float):
    df = df.sort_values("FL_DATE")
    cut = int((1.0 - eval_size) * len(df))
    return df.iloc[:cut], df.iloc[cut:]

def plot_curves(y_true, proba, out_dir, prefix):
    fpr, tpr, _ = roc_curve(y_true, proba)
    roc_auc = roc_auc_score(y_true, proba)
    plt.figure(figsize=(5,4))
    plt.plot(fpr, tpr, lw=2); plt.plot([0,1],[0,1],'--', lw=1)
    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title(f"ROC (AUC={roc_auc:.3f})")
    plt.tight_layout(); plt.savefig(os.path.join(out_dir, f"{prefix}_roc.png")); plt.close()

    p, r, _ = precision_recall_curve(y_true, proba)
    pr_auc = average_precision_score(y_true, proba)
    plt.figure(figsize=(5,4))
    plt.plot(r, p, lw=2)
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title(f"PR (AP={pr_auc:.3f})")
    plt.tight_layout(); plt.savefig(os.path.join(out_dir, f"{prefix}_pr.png")); plt.close()

def make_ohe():
    try:
        return OneHotEncoder(handle_unknown="ignore", min_frequency=0.01, sparse_output=True)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", min_frequency=0.01, sparse=True)

def main():
    ap = argparse.ArgumentParser(description="Train LightGBM and CatBoost baselines.")
    ap.add_argument("--in_path", type=str, default="data/processed/flights_with_weather.parquet")
    ap.add_argument("--out_dir", type=str, default="models")
    ap.add_argument("--eval_size", type=float, default=0.2)
    ap.add_argument("--sample_frac", type=float, default=1.0)
    ap.add_argument("--split", type=str, choices=["time","random"], default="time")
    ap.add_argument("--lgb_estimators", type=int, default=1000)
    ap.add_argument("--cat_iters", type=int, default=1000)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    df = coerce(read_parquet(args.in_path))
    if args.sample_frac < 1.0:
        df = df.sample(frac=args.sample_frac, random_state=RANDOM_STATE)

    X_all = df[CATEGORICAL + NUMERIC + (["FL_DATE"] if "FL_DATE" in df.columns else [])].copy()
    y_all = df["is_delayed"].values

    if args.split == "time" and "FL_DATE" in X_all.columns:
        tmp = X_all.join(pd.Series(y_all, name="y"))
        train_df, valid_df = time_split(tmp, args.eval_size)
        X_train = train_df.drop(columns=["y", "FL_DATE"], errors="ignore")
        y_train = train_df["y"].values
        X_valid = valid_df.drop(columns=["y", "FL_DATE"], errors="ignore")
        y_valid = valid_df["y"].values
    else:
        X = X_all.drop(columns=["FL_DATE"], errors="ignore")
        X_train, X_valid, y_train, y_valid = train_test_split(
            X, y_all, test_size=args.eval_size, random_state=RANDOM_STATE, stratify=y_all
        )

    pre = ColumnTransformer([
        ("cat", make_ohe(), CATEGORICAL),
        ("num", "passthrough", NUMERIC)
    ])

    # ----- LightGBM -----
    lgb = LGBMClassifier(
        n_estimators=args.lgb_estimators,
        learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    pipe_lgb = Pipeline([("pre", pre), ("clf", lgb)])
    Xtr = pipe_lgb.named_steps["pre"].fit_transform(X_train)
    Xva = pipe_lgb.named_steps["pre"].transform(X_valid)
    pipe_lgb.named_steps["clf"].fit(
        Xtr, y_train,
        eval_set=[(Xva, y_valid)],
        eval_metric="auc",
        verbose=False
    )
    proba_lgb = pipe_lgb.named_steps["clf"].predict_proba(Xva)[:,1]
    metrics_lgb = {
        "roc_auc": float(roc_auc_score(y_valid, proba_lgb)),
        "pr_auc": float(average_precision_score(y_valid, proba_lgb)),
        "report": classification_report(y_valid, (proba_lgb >= 0.5).astype(int), output_dict=True)
    }
    plot_curves(y_valid, proba_lgb, args.out_dir, prefix="lgb")

    # ----- CatBoost -----
    cat = CatBoostClassifier(
        iterations=args.cat_iters,
        learning_rate=0.05,
        depth=8,
        loss_function="Logloss",
        eval_metric="AUC",
        auto_class_weights="Balanced",
        random_state=RANDOM_STATE,
        verbose=False
    )
    pipe_cat = Pipeline([("pre", pre), ("clf", cat)])
    Xtr_c = pipe_cat.named_steps["pre"].fit_transform(X_train)
    Xva_c = pipe_cat.named_steps["pre"].transform(X_valid)
    pipe_cat.named_steps["clf"].fit(Xtr_c, y_train, eval_set=[(Xva_c, y_valid)], verbose=False)
    proba_cat = pipe_cat.named_steps["clf"].predict_proba(Xva_c)[:,1]
    metrics_cat = {
        "roc_auc": float(roc_auc_score(y_valid, proba_cat)),
        "pr_auc": float(average_precision_score(y_valid, proba_cat)),
        "report": classification_report(y_valid, (proba_cat >= 0.5).astype(int), output_dict=True)
    }
    plot_curves(y_valid, proba_cat, args.out_dir, prefix="cat")

    # Persist
    joblib.dump(pipe_lgb, os.path.join(args.out_dir, "lightgbm_model.joblib"))
    joblib.dump(pipe_cat, os.path.join(args.out_dir, "catboost_model.joblib"))
    with open(os.path.join(args.out_dir, "baseline_metrics.json"), "w") as f:
        json.dump({"lightgbm": metrics_lgb, "catboost": metrics_cat}, f, indent=2)

    print(f"LGB  ROC-AUC={metrics_lgb['roc_auc']:.3f}  PR-AUC={metrics_lgb['pr_auc']:.3f}")
    print(f"CAT  ROC-AUC={metrics_cat['roc_auc']:.3f}  PR-AUC={metrics_cat['pr_auc']:.3f}")

if __name__ == "__main__":
    main()
