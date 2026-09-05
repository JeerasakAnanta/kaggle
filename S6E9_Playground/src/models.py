"""Model zoo. Only deps already in env: scikit-learn + catboost."""
from catboost import CatBoostClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression


def make_models(random_state: int = 42) -> dict:
    return {
        "logreg": LogisticRegression(
            max_iter=2000, class_weight="balanced", C=0.5, random_state=random_state
        ),
        "hgb": HistGradientBoostingClassifier(
            max_iter=600,
            learning_rate=0.05,
            max_leaf_nodes=63,
            min_samples_leaf=50,
            l2_regularization=5.0,
            early_stopping=True,
            validation_fraction=0.1,
            random_state=random_state,
        ),
        "rf": RandomForestClassifier(
            n_estimators=500,
            min_samples_leaf=5,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=random_state,
        ),
        "catboost": CatBoostClassifier(
            iterations=2500,
            learning_rate=0.04,
            depth=7,
            l2_leaf_reg=5.0,
            auto_class_weights="Balanced",
            eval_metric="AUC",
            random_seed=random_state,
            verbose=False,
        ),
    }
