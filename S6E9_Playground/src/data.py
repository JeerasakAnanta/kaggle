"""I/O helpers: load raw CSVs, map target to 0/1."""
import pandas as pd

from .config import POS_LABEL, TARGET, TRAIN_PATH, TEST_PATH, ID_COL


def load_train(path=TRAIN_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


def load_test(path=TEST_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


def split_xy(df: pd.DataFrame, target: str = TARGET):
    y = (df[target] == POS_LABEL).astype(int)
    X = df.drop(columns=[target])
    return X, y


def save_submission(test_ids: pd.Series, preds, path, target: str = TARGET) -> None:
    import pathlib
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sub = pd.DataFrame({ID_COL: test_ids, target: preds})
    sub.to_csv(path, index=False)
    print(f"saved -> {path}  shape={sub.shape}")
