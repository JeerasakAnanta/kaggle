"""Stratified K-Fold CV with ROC-AUC. Saves OOF preds + per-fold scores.

Usage:
    python -m src.train --model catboost --n-splits 5
    python -m src.train --model hgb --n-splits 5
"""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from .config import ID_COL, MODEL_DIR, N_SPLITS, OUTPUT_DIR, RANDOM_STATE, TARGET
from .data import load_train, split_xy
from .features import FeatureEngineer, full_feature_pipeline
from .models import make_models

CAT_FEATURES_RAW = [
    "Gender", "City_Type", "Current_Car_Type",
    "Home_Charging_Possible", "Subsidy_Available", "Range_Anxiety_Level",
]


def run_cv(model_name: str, n_splits: int = N_SPLITS, seed: int = RANDOM_STATE):
    df = load_train()
    X, y = split_xy(df)
    ids = df[ID_COL].to_numpy()
    X = X.drop(columns=[ID_COL])  # id is a row key only — never a feature

    models = make_models(seed)
    if model_name not in models:
        raise ValueError(f"unknown model {model_name}, choose from {list(models)}")

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof = np.zeros(len(X))
    scores = []

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
        X_tr, X_va = X.iloc[tr_idx].copy(), X.iloc[va_idx].copy()
        y_tr, y_va = y.iloc[tr_idx].to_numpy(), y.iloc[va_idx].to_numpy()

        if model_name == "catboost":
            # Native categorical handling on raw + engineered frame
            fe = FeatureEngineer()
            X_tr_f = fe.fit_transform(X_tr)
            X_va_f = fe.transform(X_va)
            cat_idx = [X_tr_f.columns.get_loc(c) for c in CAT_FEATURES_RAW]
            model = make_models(seed)[model_name]
            model.fit(X_tr_f, y_tr, cat_features=cat_idx,
                      eval_set=(X_va_f, y_va), use_best_model=True)
            pred = model.predict_proba(X_va_f)[:, 1]
            fold_model = {"fe": fe, "model": model, "native_cat": True}
        else:
            use_scaling = model_name in ("logreg",)
            pipe = full_feature_pipeline(make_models(seed)[model_name], use_scaling=use_scaling)
            pipe.fit(X_tr, y_tr)
            pred = pipe.predict_proba(X_va)[:, 1]
            fold_model = pipe

        auc = roc_auc_score(y_va, pred)
        scores.append(auc)
        oof[va_idx] = pred
        print(f"[{model_name}] fold {fold}: AUC={auc:.5f}", flush=True)

        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(fold_model, MODEL_DIR / f"{model_name}_fold{fold}.pkl")

    overall = roc_auc_score(y, oof)
    print(f"[{model_name}] mean AUC={np.mean(scores):.5f} ± {np.std(scores):.5f} | OOF AUC={overall:.5f}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({ID_COL: ids, f"oof_{model_name}": oof, TARGET: y.to_numpy()}).to_csv(
        OUTPUT_DIR / f"oof_{model_name}.csv", index=False
    )
    with open(OUTPUT_DIR / f"cv_{model_name}.json", "w") as f:
        json.dump({"model": model_name, "fold_auc": scores,
                   "mean": float(np.mean(scores)), "std": float(np.std(scores)),
                   "oof_auc": float(overall)}, f, indent=2)
    return scores, overall


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="catboost", choices=list(make_models().keys()))
    p.add_argument("--n-splits", type=int, default=N_SPLITS)
    p.add_argument("--seed", type=int, default=RANDOM_STATE)
    args = p.parse_args()
    run_cv(args.model, args.n_splits, args.seed)


if __name__ == "__main__":
    main()
