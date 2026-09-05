"""Inference: average fold models -> submission.csv.

Usage:
    python -m src.predict --models catboost hgb --output outputs/submissions/submission.csv
    # single fast baseline:
    python -m src.predict --models hgb
"""
import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .config import ID_COL, MODEL_DIR, SUBMISSION_DIR, TARGET
from .data import load_test, save_submission


def predict_with_folds(model_name: str, X: pd.DataFrame) -> np.ndarray:
    files = sorted(MODEL_DIR.glob(f"{model_name}_fold*.pkl"))
    if not files:
        raise FileNotFoundError(f"no trained folds for {model_name} in {MODEL_DIR} — run train first")
    preds = []
    for f in files:
        obj = joblib.load(f)
        if isinstance(obj, dict) and obj.get("native_cat"):  # catboost native path
            Xt = obj["fe"].transform(X)
            preds.append(obj["model"].predict_proba(Xt)[:, 1])
        else:  # sklearn pipeline
            preds.append(obj.predict_proba(X)[:, 1])
    return float(np.mean(preds, axis=0).mean()), np.mean(preds, axis=0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", default=["catboost"])
    p.add_argument("--weights", nargs="*", type=float, default=None)
    p.add_argument("--output", default=str(SUBMISSION_DIR / "submission.csv"))
    args = p.parse_args()

    test = load_test()
    X_test = test.drop(columns=[ID_COL])

    all_preds, means = [], []
    for m in args.models:
        mean_p, p_fold = predict_with_folds(m, X_test)
        all_preds.append(p_fold)
        means.append(mean_p)
        print(f"{m}: mean_pred={mean_p:.5f}")

    w = np.array(args.weights) if args.weights else np.ones(len(all_preds)) / len(all_preds)
    w = w / w.sum()
    final = np.average(np.vstack(all_preds), axis=0, weights=w)
    print(f"ensemble weights={dict(zip(args.models, w.round(3)))} mean={final.mean():.5f}")

    # Safety clip (AUC doesn't need it, but keeps file sane)
    final = np.clip(final, 1e-4, 1 - 1e-4)
    save_submission(test[ID_COL], final, args.output)


if __name__ == "__main__":
    main()
