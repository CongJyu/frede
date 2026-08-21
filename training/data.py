"""Dataset loading, cleaning and splitting (port of notebook Phase 1)."""
from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split

from app.preprocess import preprocess
from . import config as tc


def load_yelp(sample_n: int | None = None) -> pd.DataFrame:
    """Load the Yelp train split, sample rows, and binarize stars→fake/real.

    Labels: 1–2★ → fake (1), 4–5★ → real (0); 3★ (neutral) is dropped.
    """
    sample_n = sample_n or tc.SAMPLE_N
    try:
        import datasets

        ds = datasets.load_dataset(
            "Yelp/yelp_review_full", split="train", cache_dir=str(tc.HF_CACHE_DIR)
        )
        df = ds.to_pandas()
    except Exception as exc:  # noqa: BLE001
        # Fallback: stream and take enough rows to sample from.
        print(f"[datasets full load failed ({exc}); falling back to streaming]")
        import datasets

        ds = datasets.load_dataset(
            "Yelp/yelp_review_full",
            split="train",
            streaming=True,
            cache_dir=str(tc.HF_CACHE_DIR),
        )
        rows = []
        for i, row in enumerate(ds):
            if i >= sample_n * 2:
                break
            rows.append(row)
        df = pd.DataFrame(rows)

    df = df.sample(n=min(sample_n, len(df)), random_state=tc.RANDOM_SEED)
    df = df.rename(columns={"label": "stars"})
    df = df[df["stars"] != 2].copy()  # drop 3-star
    df["label"] = (df["stars"] <= 1).astype(int)
    df = df[["text", "label"]].reset_index(drop=True)
    return df


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Dedupe, drop nulls, and apply the SAME preprocessing used at serve time."""
    before = len(df)
    df = df.drop_duplicates(subset=["text"])
    df = df.dropna(subset=["text", "label"])
    df["clean_text"] = df["text"].astype(str).apply(preprocess)
    df["label"] = df["label"].astype(int)
    print(f"Cleaned: dropped {before - len(df)} rows; {len(df)} remain.")
    return df


def make_splits(df: pd.DataFrame):
    """Stratified 70/15/15 train/val/test split (matches the notebook)."""
    X = df["clean_text"].tolist()
    y = df["label"].tolist()
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=tc.VAL_FRAC, random_state=tc.RANDOM_SEED, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=tc.TEST_FRAC_OF_TEMP,
        random_state=tc.RANDOM_SEED, stratify=y_temp,
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def save_splits(X_train, X_val, X_test, y_train, y_val, y_test) -> None:
    tc.DATA_DIR.mkdir(parents=True, exist_ok=True)
    split_df = pd.DataFrame(
        {
            "text": X_train + X_val + X_test,
            "label": y_train + y_val + y_test,
            "split": (
                    ["train"] * len(X_train)
                    + ["val"] * len(X_val)
                    + ["test"] * len(X_test)
            ),
        }
    )
    path = tc.DATA_DIR / "splits.csv"
    split_df.to_csv(path, index=False)
    print(f"Splits saved -> {path}")
