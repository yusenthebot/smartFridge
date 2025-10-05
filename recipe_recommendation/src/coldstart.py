import os
import ast
import json
import random
import pandas as pd
import numpy as np
from tqdm import tqdm
import warnings

from .candidate import coarse_rank_candidates, hard_filter, rule_generate_candidates
from .feature import build_features
from .io import load_recipes_csv, load_ingredient_map

RECIPES_PATH = load_recipes_csv()
INGREDIENT_MAP = load_ingredient_map()
PARENTS = INGREDIENT_MAP["parents"]
CHILDREN = INGREDIENT_MAP["children"]

def parse_list(x):
    """Convert a stringified list into a Python list safely."""
    if pd.isna(x) or x == "":
        return []
    if isinstance(x, list):
        return x
    try:
        return ast.literal_eval(x)
    except Exception:
        return []

def parse_set(x):
    """Convert a stringified collection into a Python set safely."""
    if pd.isna(x) or x == "":
        return set()
    if isinstance(x, set):
        return x
    if isinstance(x, (list, tuple)):
        return set(x)
    if isinstance(x, str):
        try:
            v = ast.literal_eval(x)
            if isinstance(v, (list, tuple, set)):
                return set(v)
            return {v}
        except Exception:
            return {x.strip()}
    return {x}

def _parents_pool_from_df(df: pd.DataFrame):
    cols = ["main_parent", "staple_parent", "other_parent", "seasoning_parent"]
    pool = set()
    for c in cols:
        if c in df.columns:
            for s in df[c]:
                pool |= set(s) if isinstance(s, (set, list, tuple)) else set()
    return sorted(pool)


def sample_user_parents(parents_pool,
                        user_profile=None,
                        prev_inventory=None,
                        min_items=3, max_items=10,
                        keep_ratio=0.6, reset_interval=20, round_idx=0):
    liked = set((user_profile or {}).get("other_preferences", {}).get("preferred_main", []))
    disliked = set((user_profile or {}).get("other_preferences", {}).get("disliked_main", []))
    forbidden = set((user_profile or {}).get("forbidden_parents", [])) | disliked

    pool, weights = [], []
    for p in parents_pool:
        if p in forbidden:
            continue
        w = 3.0 if p in liked else 1.0
        pool.append(p); weights.append(w)
    if not pool:
        pool, weights = parents_pool[:], [1.0] * len(parents_pool)

    inventory = set()
    force_reset = (round_idx % reset_interval == 0)
    if prev_inventory and not force_reset:
        prev_list = list(prev_inventory); random.shuffle(prev_list)
        keep_k = max(0, int(len(prev_list) * keep_ratio))
        inventory |= set(prev_list[:keep_k])

    k = random.randint(min_items, max_items)
    remain = max(0, k - len(inventory))
    for _ in range(min(remain, len(pool))):
        idx = random.choices(range(len(pool)), weights=weights, k=1)[0]
        inventory.add(pool[idx])
    return list(inventory)


def _weighted_pick3(indexes, scores, temperature=1.0):
    idxs = list(indexes)
    scs = np.array(scores, dtype=float)
    if np.any(scs < 0):
        scs = scs - scs.min()
    if scs.sum() == 0:
        scs = np.ones_like(scs)
    picks = []
    for _ in range(min(3, len(idxs))):
        probs = np.exp(scs / max(temperature, 1e-6))
        probs = probs / probs.sum()
        choice = np.random.choice(len(idxs), p=probs)
        picks.append(idxs[choice])
        idxs.pop(choice)
        scs = np.delete(scs, choice)
        if len(idxs) == 0:
            break
    return picks


# ---------- Main cold-start ----------
# ---------- Main cold-start ----------
def cold_start_ranker(user_id: str,
                      n_rounds: int = 10000,
                      topn_coarse: int = 5000,
                      topk_rule: int = 5,
                      batch_size: int = 5000,
                      switch_interval: int = 100):
    """
    Cold-start data generation for learning-to-rank.
    Top-5 selection prioritizes user pantry coverage deterministically:
    1. Fully covered recipes first (missing_count == 0)
    2. Then few missing (esp. staple/other)
    3. Heavy penalty for missing main ingredients.
    """
    base_dir = os.path.join("user_data", user_id)
    os.makedirs(base_dir, exist_ok=True)
    profile_path  = os.path.join(base_dir, "user_profile.json")
    features_path = os.path.join(base_dir, "user_features_rank.csv")

    if os.path.exists(features_path):
        print(f"[cold_start] Features already exist at {features_path}")
        return features_path

    with open(profile_path, "r", encoding="utf-8") as f:
        user_profile = json.load(f)

    # Load and parse recipes
    df_all = pd.read_csv(RECIPES_PATH)
    to_set = ["main_parent", "staple_parent", "other_parent", "seasoning_parent", "cuisine_attr"]
    to_list = ["ingredients"]
    for c in to_set:
        if c in df_all.columns:
            df_all[c] = df_all[c].apply(parse_set)
    for c in to_list:
        if c in df_all.columns:
            df_all[c] = df_all[c].apply(parse_list)

    # Step 1 hard filter
    if hard_filter is not None:
        try:
            before = len(df_all)
            mask = df_all.apply(lambda r: hard_filter(r.to_dict(), user_profile), axis=1)
            df_all = df_all[mask]
            after = len(df_all)
            print(f"[cold_start] Step1 hard filter applied: {before} -> {after}")
        except Exception as e:
            warnings.warn(f"[cold_start] hard_filter failed, skip. err={e}")

    n_chunks = (len(df_all) // batch_size) + 1
    chunks = np.array_split(df_all, n_chunks)
    parents_pool = _parents_pool_from_df(df_all)
    rows = []
    prev_inventory = None

    for i in tqdm(range(n_rounds), desc="Cold-start rounds"):
        chunk_id = (i // switch_interval) % n_chunks
        df_chunk = chunks[chunk_id].copy()

        # pantry sampling
        user_parents = sample_user_parents(
            parents_pool,
            user_profile=user_profile,
            prev_inventory=prev_inventory,
            round_idx=i
        )
        prev_inventory = user_parents

        # Step 2: coarse recall
        coarse_list = coarse_rank_candidates(
            recipes=df_chunk.to_dict(orient="records"),
            user_parents=user_parents,
            user_profile=user_profile,
            top_n=min(topn_coarse, len(df_chunk))
        )
        if not coarse_list:
            continue

        coarse_df = pd.DataFrame(coarse_list)

        # Step 3: rule rerank → Top-5 candidates (just for selecting the 5)
        rule_df = rule_generate_candidates(
            coarse_df,
            user_parents=user_parents,
            user_profile=user_profile
        )
        if rule_df.empty or len(rule_df) < topk_rule:
            continue

        top5 = rule_df.head(topk_rule).copy()

        # ===== New deterministic scoring with main priority =====
        user_set = set(user_parents)
        weighted_scores = []
        for idx, row in top5.iterrows():
            main_set   = set(row.get("main_parent", set()))
            staple_set = set(row.get("staple_parent", set()))
            other_set  = set(row.get("other_parent", set()))

            main_missing   = len(main_set   - user_set)
            staple_missing = len(staple_set - user_set)
            other_missing  = len(other_set  - user_set)

            weighted_missing = 10 * main_missing + 2 * staple_missing + 1 * other_missing
            total_missing = main_missing + staple_missing + other_missing

            weighted_scores.append((idx, weighted_missing, total_missing))

        sorted_pairs = sorted(weighted_scores, key=lambda x: (x[1], x[2]))
        picked_idxs = [idx for idx, _, _ in sorted_pairs[:3]]

        # relevance 3 / 2 / 1
        labels = {idx: 0 for idx in top5.index}
        if len(picked_idxs) > 0:
            labels[picked_idxs[0]] = 3
        if len(picked_idxs) > 1:
            labels[picked_idxs[1]] = 2
        if len(picked_idxs) > 2:
            labels[picked_idxs[2]] = 1

        # build features for all 5 candidates
        for idx, row in top5.iterrows():
            up = set(user_parents)
            main_set   = set(row.get("main_parent", set()))
            staple_set = set(row.get("staple_parent", set()))
            other_set  = set(row.get("other_parent", set()))

            recipe_dict = {
                "main": main_set,
                "staple": staple_set,
                "other": other_set,
                "seasoning": set(row.get("seasoning_parent", set())),
                "matched_main":   len(main_set   & up),
                "matched_staple": len(staple_set & up),
                "matched_other":  len(other_set  & up),
                "calories": row.get("calories", 0),
                "protein":  row.get("protein", 0),
                "fat":      row.get("fat", 0),
                "region": row.get("region", ""),
                "cuisine_attr": row.get("cuisine_attr", []),
                "ingredients": row.get("ingredients", []),
                "minutes": row.get("minutes", None),
            }

            feats = build_features(recipe_dict, user_profile)
            feats["relevance"] = float(labels[idx])
            feats["qid"] = int(i)
            rows.append(feats)

    out = pd.DataFrame(rows)
    valid_qids = out.groupby("qid").size()
    keep_qids = valid_qids[valid_qids > 1].index
    out = out[out["qid"].isin(keep_qids)].reset_index(drop=True)

    out_path = os.path.join("user_data", user_id, "user_features_rank.csv")
    out.to_csv(out_path, index=False)
    print(f"[cold_start] Saved {len(out)} rows to {out_path}")
    return out_path


if __name__ == "__main__":
    cold_start_ranker(
        user_id="user_1",
        n_rounds=10000,
        topn_coarse=20000,
        topk_rule=5,
        coverage_penalty=0.15,
        temperature=0.5
    )