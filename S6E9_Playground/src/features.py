"""Feature engineering + preprocessing.

Design notes from EDA:
- Subsidy_Available / Range_Anxiety_Level / Home_Charging_Possible are the
  strongest signals; Environmental_Concern_Level + Annual_Income_USD next.
- Gender / City_Type / Current_Car_Type are weak but still useful in trees.
- No missing values, so no imputation needed (SimpleImputer kept as safety).
- Engineered features: charging total/max, income x concern, commute x cars,
  subsidy x anxiety interaction (the two dominant categoricals).
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import CAT_FEATURES, NUM_FEATURES


class FeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X["Charging_Total"] = X["Charging_Stations_Near_Home"] + X["Charging_Stations_Near_Work"]
        X["Charging_Max"] = X[["Charging_Stations_Near_Home", "Charging_Stations_Near_Work"]].max(axis=1)
        X["Charging_Home_x_Work"] = X["Charging_Stations_Near_Home"] * X["Charging_Stations_Near_Work"]
        X["Income_x_Concern"] = X["Annual_Income_USD"] * X["Environmental_Concern_Level"]
        X["Income_per_Car"] = X["Annual_Income_USD"] / X["Number_of_Cars_Owned"].clip(lower=1)
        X["Commute_x_Cars"] = X["Daily_Commute_km"] * X["Number_of_Cars_Owned"]
        X["Subsidy_x_HomeCharge"] = (
            (X["Subsidy_Available"] == "Yes").astype(int)
            * (X["Home_Charging_Possible"] == "Yes").astype(int)
        )
        X["NoSubsidy_HighAnxiety"] = (
            (X["Subsidy_Available"] == "No").astype(int)
            | (X["Range_Anxiety_Level"] == "High").astype(int)
        ).astype(int)
        X["LowAnxiety_Subsidy"] = (
            (X["Range_Anxiety_Level"] == "Low").astype(int)
            & (X["Subsidy_Available"] == "Yes").astype(int)
        ).astype(int)
        return X


ENGINEERED_NUM = [
    "Charging_Total", "Charging_Max", "Charging_Home_x_Work",
    "Income_x_Concern", "Income_per_Car", "Commute_x_Cars",
    "Subsidy_x_HomeCharge", "NoSubsidy_HighAnxiety", "LowAnxiety_Subsidy",
]


def build_preprocessor(use_scaling: bool = True) -> ColumnTransformer:
    """For linear / tree models that need one-hot + scaled numerics."""
    num_all = NUM_FEATURES + ENGINEERED_NUM
    num_pipe = Pipeline([("scaler", StandardScaler())]) if use_scaling else "passthrough"
    pre = ColumnTransformer(
        [
            ("num", num_pipe, num_all),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT_FEATURES),
        ],
        verbose_feature_names_out=False,
    )
    return pre


def full_feature_pipeline(estimator, use_scaling: bool = True) -> Pipeline:
    return Pipeline(
        [
            ("feat", FeatureEngineer()),
            ("pre", build_preprocessor(use_scaling)),
            ("model", estimator),
        ]
    )
