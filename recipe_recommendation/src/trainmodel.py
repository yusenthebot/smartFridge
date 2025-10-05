import os
import joblib
import warnings
import numpy as np
import pandas as pd
from typing import List, Tuple, Sequence, Optional
from xgboost import XGBRanker
from sklearn.model_selection import train_test_split
from sklearn.metrics import ndcg_score
from pandas.api.types import is_numeric_dtype


# ----------------------------- Helpers -----------------------------
def _pick_feature_cols(df: pd.DataFrame, drop_cols: Sequence[str]) -> List[str]:
    """
    Pick numeric feature columns robustly, excluding drop_cols.
    Uses pandas is_numeric_dtype to correctly include nullable ints/floats/bools.
    """
    cols = []
    for c in df.columns:
        if c in drop_cols:
            continue
        if is_numeric_dtype(df[c]):
            cols.append(c)
    return cols


def _sort_and_pack_by_qid(
    X: pd.DataFrame, y: pd.Series, qid: pd.Series, feature_cols: List[str]
) -> Tuple[pd.DataFrame, np.ndarray, List[int], np.ndarray]:
    """
    Sort rows by qid so that group sizes match the sample order.
    Returns:
        X_sorted, y_sorted, groups, qid_sorted (aligned with X_sorted/y_sorted)
    """
    packed = X.copy()
    packed["_label"] = y.values
    packed["_qid"] = qid.values
    packed = packed.sort_values("_qid").reset_index(drop=True)

    groups = packed.groupby("_qid").size().tolist()
    X_sorted = packed[feature_cols].copy()
    y_sorted = packed["_label"].astype(float).values
    qid_sorted = packed["_qid"].values
    return X_sorted, y_sorted, groups, qid_sorted


def _eval_mean_ndcg(
    model: XGBRanker,
    X_val: pd.DataFrame,
    y_val,              # can be np.ndarray or pd.Series
    qid_val,            # aligned with X_val/y_val
    ks: Sequence[int] = (5, 10),
) -> dict:
    """
    Compute mean NDCG@k for each k in ks over validation queries.
    Accepts numpy arrays or pandas Series.
    """
    # Try to respect early-stopping best iteration if available (xgboost>=2.0)
    try:
        preds = model.predict(X_val, iteration_range=(0, model.best_iteration + 1))
    except Exception:
        preds = model.predict(X_val)

    y_arr = np.asarray(y_val)
    q_arr = np.asarray(qid_val)

    out = {}
    for k in ks:
        ndcgs = []
        for q in np.unique(q_arr):
            mask = (q_arr == q)
            if mask.sum() < 2:
                continue
            ndcgs.append(ndcg_score([y_arr[mask]], [preds[mask]], k=k))
        out[f"NDCG@{k}"] = float(np.mean(ndcgs)) if ndcgs else 0.0
    return out



# ----------------------------- Main Trainer -----------------------------
def train_model_ranker(
    user_id: str = "user_1",
    features_path: Optional[str] = None,
    save_model: bool = True,
    model_params: Optional[dict] = None,
    val_ratio: float = 0.2,
    random_state: int = 42,
    max_rows: Optional[int] = None,
):
    """
    Train an XGBoost Learning-to-Rank model (XGBRanker) on cold-start generated data.

    Expected input CSV (from cold_start.py):
      - qid:       query id (one round of pantry sampling = one query)
      - relevance: graded relevance label (e.g., 3/2/1/0)
      - features:  numeric columns produced by build_features (and any extra numeric signals)

    The function:
      1) Reads the CSV
      2) Selects numeric feature columns robustly
      3) Splits train/val by qid to avoid leakage
      4) Sorts each split by qid and builds group sizes aligned to sample order
      5) Trains XGBRanker and reports NDCG@5/10
      6) Saves model to user_data/<user_id>/ranker.pkl
    """
    base_dir = os.path.join("user_data", user_id)
    os.makedirs(base_dir, exist_ok=True)

    # Resolve features path
    if features_path is None:
        features_path = os.path.join(base_dir, "user_features_rank.csv")
    if not os.path.exists(features_path):
        raise FileNotFoundError(
            f"[train_model_ranker] Cold-start features not found at: {features_path}\n"
            f"Please run cold_start_ranker(user_id='{user_id}') first."
        )

    # Load data
    df = pd.read_csv(features_path)
    if max_rows is not None and len(df) > max_rows:
        df = df.sample(max_rows, random_state=random_state).reset_index(drop=True)

    # Basic validation
    if "qid" not in df.columns or "relevance" not in df.columns:
        raise ValueError("Input CSV must contain 'qid' and 'relevance' columns.")

    # Fill NaNs in label/qid (should not happen, but defensive)
    df["qid"] = pd.to_numeric(df["qid"], errors="coerce").fillna(-1).astype(int)
    df["relevance"] = pd.to_numeric(df["relevance"], errors="coerce").fillna(0).astype(float)

    # Pick numeric feature columns robustly
    drop_cols = {"qid", "relevance"}
    feature_cols = _pick_feature_cols(df, drop_cols)
    if not feature_cols:
        raise ValueError("No numeric feature columns found in dataset.")

    # Ensure numeric + finite values only (replace inf/nan with 0)
    df[feature_cols] = df[feature_cols].apply(pd.to_numeric, errors="coerce")
    df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # Split by qid to avoid leakage across queries
    unique_qids = df["qid"].unique()
    if len(unique_qids) < 2:
        warnings.warn("Only one unique qid found — ranking training may be ineffective.")
    train_qids, val_qids = train_test_split(
        unique_qids, test_size=val_ratio, random_state=random_state
    )
    train_mask = df["qid"].isin(train_qids)
    val_mask = df["qid"].isin(val_qids)

    # Split dataframes
    X_train_raw = df.loc[train_mask, feature_cols]
    y_train_raw = df.loc[train_mask, "relevance"]
    qid_train = df.loc[train_mask, "qid"]

    X_val_raw = df.loc[val_mask, feature_cols]
    y_val_raw = df.loc[val_mask, "relevance"]
    qid_val = df.loc[val_mask, "qid"]

    # Sort by qid and build group sizes aligned with sample order (CRITICAL for XGBRanker)
    X_train, y_train, group_train, _ = _sort_and_pack_by_qid(
    X_train_raw, y_train_raw, qid_train, feature_cols
    )
    X_val, y_val, group_val, qid_val_sorted = _sort_and_pack_by_qid(
        X_val_raw, y_val_raw, qid_val, feature_cols
    )


    print(f"[ranker] #Train groups: {len(group_train)} | #Val groups: {len(group_val)}")
    print(f"[ranker] Train rows: {len(X_train)} | Val rows: {len(X_val)} | #Features: {len(feature_cols)}")

    # Default model params
    default_params = dict(
        objective="rank:ndcg",
        eval_metric="ndcg",
        n_estimators=400,
        learning_rate=0.08,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=random_state,
        tree_method="hist",
        reg_lambda=1.0,
        reg_alpha=0.0,
    )
    if model_params:
        default_params.update(model_params)

    model = XGBRanker(**default_params)

    # Fit model (XGBRanker requires group/group for eval_set as well)
    fit_kwargs = dict(
    X=X_train,
    y=y_train,
    group=group_train,
    eval_set=[(X_val, y_val)],
    eval_group=[group_val],
    verbose=False,
)

    try:
        # Newer xgboost versions (some builds) support early_stopping_rounds on Ranker
        model.fit(early_stopping_rounds=50, **fit_kwargs)  # maximize=True is inferred by 'ndcg'
    except TypeError:
        # Fallback to callback API (older versions)
        try:
            from xgboost.callback import EarlyStopping
            model.fit(callbacks=[EarlyStopping(rounds=50, save_best=True, maximize=True)], **fit_kwargs)
        except Exception:
            # Last resort: train without early stopping
            model.fit(**fit_kwargs)

    # Evaluate mean NDCG@5/10
    metrics = _eval_mean_ndcg(model, X_val, y_val, qid_val_sorted, ks=(5, 10))

    print("[ranker] Validation metrics:", " ".join(f"{k}={v:.4f}" for k, v in metrics.items()))

    # Save model
    if save_model:
        model_path = os.path.join(base_dir, "ranker.pkl")
        joblib.dump(model, model_path)
        print(f"[ranker] Model saved to {model_path}")

    return model, metrics, feature_cols


if __name__ == "__main__":
    # Example run
    train_model_ranker(
        user_id="user_1",
        save_model=True,
        val_ratio=0.2,
        random_state=42,
        max_rows=None,  # or set an upper bound for quick iterations, e.g., 200_000
        model_params=None,  # override defaults if desired
    )
