# main.py
# -*- coding: utf-8 -*-
"""
Entry point for the new pipeline:
1) I/O init & parsing
2) Load user parents from recipe_input.json via ingredient_map (children -> parent)
3) Ensure cold-start features & trained ranker exist
4) Step 2: Coarse ranking
5) Step 3: ML reranking
6) Pretty print top results
"""

import os
import json
import ast
import pandas as pd
from pathlib import Path
import shutil

from recipe_recommendation.src.io import load_recipes_csv, load_ingredient_map, download_file
from recipe_recommendation.src.coldstart import cold_start_ranker
from recipe_recommendation.src.trainmodel import train_model_ranker
from recipe_recommendation.src.candidate import (
    coarse_rank_candidates,
    ml_generate_candidates,
    hard_filter,
)
from recipe_recommendation.src.highlight import (
    print_candidates,
    diversify_topk_with_min_clusters,
)
from recipe_recommendation.src.feature import build_features, build_cluster_features
from recipe_recommendation.src.embedding import find_most_similar_user


BASE_DIR = Path(__file__).resolve().parent
USER_DATA_DIR = BASE_DIR / "user_data"



def load_recipes() -> pd.DataFrame:
    """
    Load recipes.csv as DataFrame and assign a unique recipe_id to each row.
    This keeps io.py focused on downloading only.
    """
    path = download_file("recipes.csv")
    df = pd.read_csv(path)
    df.reset_index(drop=True, inplace=True)
    df["recipe_id"] = df.index
    return df

# ---------------------------
# Helpers: parsing utilities
# ---------------------------
def parse_list(x):
    """Parse a cell into Python list; tolerant to str/NaN/set."""
    if isinstance(x, list):
        return x
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return []
    if isinstance(x, set):
        return list(x)
    s = str(x).strip()
    if not s:
        return []
    # Try literal eval first
    try:
        v = ast.literal_eval(s)
        if isinstance(v, list):
            return v
        if isinstance(v, set):
            return list(v)
    except Exception:
        pass
    # Fallback: comma-separated
    s = s.strip("[]")
    parts = [t.strip() for t in s.split(",") if t.strip()]
    return parts


def parse_set(x):
    """Parse a cell into Python set via parse_list."""
    return set(parse_list(x))


# -------------------------------------
# Map user CV result -> parent set
# -------------------------------------
def load_user_parents_from_json(json_path, ingredient_map, conf_th=0.8):
    """
    Map raw ingredient names to parent categories using ingredient_map["children"].
    If a name is already a parent in ingredient_map["parents"], keep it.
    Unknown terms are skipped.
    """
    parents_map = ingredient_map.get("parents", {}) or {}
    children_map = ingredient_map.get("children", {}) or {}

    if not os.path.exists(json_path):
        raise FileNotFoundError(f"recipe_input.json not found at: {json_path}")

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    out = []
    hi, lo = [], []
    for ing in data.get("ingredients", []):
        name = (ing.get("name") or "").strip().lower().replace("_", " ")
        conf = float(ing.get("confidence", 0.0))
        parent = None
        if name in children_map:
            # Prefer "parent" field; fall back to "fallback" if present
            parent = children_map[name].get("parent") or children_map[name].get("fallback")
        elif name in parents_map:
            parent = name

        if parent and conf >= conf_th:
            out.append(parent)
            hi.append((name, parent))
        else:
            lo.append(name)

    if hi:
        print("High-confidence ingredients mapped to parents:")
        for child, p in hi:
            print(f"  - {child} → {p}")
    if lo:
        print(f"Ignored (low confidence or no parent found): {sorted(set(lo))}")

    return sorted(set(out))


def normalize_user_profile(profile):
    """Fill missing keys and set defaults to avoid None errors downstream."""
    # Diet
    diet = profile.get("diet", {})
    profile["diet"] = {"vegetarian_type": diet.get("vegetarian_type", "flexible")}

    # Allergies
    if "allergies" not in profile or profile["allergies"] is None:
        profile["allergies"] = []

    # Region
    if "region_preference" not in profile or profile["region_preference"] is None:
        profile["region_preference"] = []

    # Nutritional goals
    if "nutritional_goals" not in profile or profile["nutritional_goals"] is None:
        profile["nutritional_goals"] = {
            "calories": {"min": 0, "max": 9999},
            "protein": {"min": 0, "max": 999}
        }
    else:
        ng = profile["nutritional_goals"]
        ng["calories"] = ng.get("calories", {"min": 0, "max": 9999})
        ng["protein"] = ng.get("protein", {"min": 0, "max": 999})

    # Other preferences
    other = profile.get("other_preferences", {})
    if not other:
        other = {}
    other["preferred_main"] = other.get("preferred_main", [])
    other["disliked_main"] = other.get("disliked_main", [])
    other["cooking_time_max"] = other.get("cooking_time_max", None)
    profile["other_preferences"] = other

    return profile

def is_profile_empty(profile):
    """Return True if the profile has almost no meaningful preferences."""
    if profile.get("diet", {}).get("vegetarian_type") not in [None, "", "flexible"]:
        return False
    if profile.get("allergies"):
        return False
    if profile.get("region_preference"):
        return False

    ng = profile.get("nutritional_goals", {})
    if ng.get("calories") or ng.get("protein"):
        c = ng.get("calories", {})
        p = ng.get("protein", {})
        if c.get("min", 0) > 0 or c.get("max", 0) < 9999:
            return False
        if p.get("min", 0) > 0 or p.get("max", 0) < 999:
            return False

    other = profile.get("other_preferences", {})
    if other.get("preferred_main") or other.get("disliked_main") or other.get("cooking_time_max"):
        return False

    return True

def fill_default_preferences(profile):
    """
    Fill some lightweight, neutral defaults so that hard_filter and cold_start
    can work efficiently even for new users with no explicit preferences.
    """
    profile["diet"]["vegetarian_type"] = "flexible"
    profile["region_preference"] = ["North America", "Europe"]
    profile["nutritional_goals"]["protein"] = {"min": 50, "max": 150}
    profile["nutritional_goals"]["calories"] = {"min": 400, "max": 2000}
    profile["other_preferences"]["cooking_time_max"] = 45
    return profile

def ensure_user_profile(user_id):
    """
    Load user profile JSON, normalize structure, and fill default preferences
    if the profile is empty. This ensures downstream code never breaks on None
    and avoids extremely slow cold start for users with no preferences.
    """
    import os, json

    profile_file = USER_DATA_DIR / user_id / "user_profile.json"
    if not os.path.exists(profile_file):
        raise FileNotFoundError(
            f"Missing profile: {profile_file}. Please create one first."
        )

    # Load profile
    with open(profile_file, "r", encoding="utf-8") as f:
        profile = json.load(f)
    # Normalize structure
    profile = normalize_user_profile(profile)
    # Detect if almost empty
    if is_profile_empty(profile):
        print(f"[profile] User {user_id} has an empty or near-empty profile. Filling defaults...")
        profile = fill_default_preferences(profile)

    return profile

    
def save_user_profile(user_id, profile):
    profile_path = USER_DATA_DIR / user_id / "user_profile.json"
    profile_path.parent.mkdir(parents=True, exist_ok=True)
    with open(profile_path, "w", encoding="utf-8") as f:
        json.dump(profile, f, indent=2)

def collect_user_feedback(user_id: str, selected_recipe_row: dict, user_profile: dict, qid: int):
    """
    Collect a single feedback sample.
    - Uses build_features() to ensure feature alignment with training
    - Maintains a fixed feature order via feature_order.json
    """
    user_dir = USER_DATA_DIR / user_id
    user_dir.mkdir(parents=True, exist_ok=True)
    feedback_path = user_dir / "feedback.csv"
    feature_order_path = user_dir / "feature_order.json"

    recipe_dict = {
        "main": selected_recipe_row.get("main_parent", set()),
        "staple": selected_recipe_row.get("staple_parent", set()),
        "other": selected_recipe_row.get("other_parent", set()),
        "seasoning": selected_recipe_row.get("seasoning_parent", set()),
        "matched_main": len(selected_recipe_row.get("main_parent", set()) & set(user_profile.get("user_parents", []))),
        "matched_staple": len(selected_recipe_row.get("staple_parent", set()) & set(user_profile.get("user_parents", []))),
        "matched_other": len(selected_recipe_row.get("other_parent", set()) & set(user_profile.get("user_parents", []))),
        "calories": selected_recipe_row.get("calories", 0),
        "protein": selected_recipe_row.get("protein", 0),
        "fat": selected_recipe_row.get("fat", 0),
        "region": selected_recipe_row.get("region", ""),
        "cuisine_attr": selected_recipe_row.get("cuisine_attr", []),
        "ingredients": selected_recipe_row.get("ingredients", []),
        "minutes": selected_recipe_row.get("minutes", None),
    }
    features = build_features(recipe_dict, user_profile)

    if os.path.exists(feature_order_path):
        with open(feature_order_path, "r", encoding="utf-8") as f:
            feature_order = json.load(f)
    else:
        feature_order = list(features.keys())
        with open(feature_order_path, "w", encoding="utf-8") as f:
            json.dump(feature_order, f, indent=2)

    for feat in features.keys():
        if feat not in feature_order:
            feature_order.append(feat)
            with open(feature_order_path, "w", encoding="utf-8") as f:
                json.dump(feature_order, f, indent=2)

    row_data = {feat: features.get(feat, 0) for feat in feature_order}
    row_data["recipe_id"] = selected_recipe_row["recipe_id"]
    row_data["qid"] = qid
    row_data["relevance"] = 5

    new_row_df = pd.DataFrame([row_data])

    if os.path.exists(feedback_path):
        old_df = pd.read_csv(feedback_path)
        for col in new_row_df.columns:
            if col not in old_df.columns:
                old_df[col] = 0
        for col in old_df.columns:
            if col not in new_row_df.columns:
                new_row_df[col] = 0
        df = pd.concat([old_df, new_row_df], ignore_index=True)
    else:
        df = new_row_df
    df.to_csv(feedback_path, index=False)
    print(f"[feedback] Saved user feedback to {feedback_path} ({len(df)} rows total)")


def ensure_model(user_id):
    base_dir = USER_DATA_DIR / user_id
    base_dir.mkdir(parents=True, exist_ok=True)
    features_rank = base_dir / "user_features_rank.csv"
    model_file = base_dir / "ranker.pkl"

    if not os.path.exists(features_rank):
        print("[main] No cold-start features found; running cold_start_ranker() ...")
        cold_start_ranker(user_id=user_id)

    if not os.path.exists(model_file):
        print("[main] No model found; training ranker with train_model_ranker() ...")
        train_model_ranker(user_id=user_id)

    return model_file


def prepare_recipes_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize key columns to list/set shapes that our candidate/feature modules expect.
    """
    df = df.copy()

    # list-like columns
    for col in ["staple", "main", "seasoning", "other", "ingredients"]:
        if col in df.columns:
            df[col] = df[col].apply(parse_list)

    # set-like columns
    for col in ["staple_parent", "main_parent", "seasoning_parent", "other_parent", "cuisine_attr"]:
        if col in df.columns:
            df[col] = df[col].apply(parse_set)

    # region: allow str or set; if it looks like list/set, cast to set; otherwise keep str
    if "region" in df.columns:
        def _region_norm(x):
            if isinstance(x, (set, list)):
                return set(x)
            try:
                v = ast.literal_eval(str(x))
                if isinstance(v, (set, list)):
                    return set(v)
            except Exception:
                pass
            return str(x) if pd.notna(x) else ""
        df["region"] = df["region"].apply(_region_norm)

    return df


def maybe_retrain_model(user_id):
    profile_path = USER_DATA_DIR / user_id / "user_profile.json"
    if not profile_path.exists():
        return

    profile = json.loads(profile_path.read_text())
    n_fb = profile.get("num_feedback", 0)

    if n_fb > 0 and n_fb % 20 == 0:
        print(f"[main] {n_fb} feedback reached, retraining ranker...")

        model_path = USER_DATA_DIR / user_id / "ranker.pkl"
        if model_path.exists():
            model_path.unlink()

        train_model_ranker(user_id)

def get_next_qid(user_id: str) -> int:
    user_dir = USER_DATA_DIR / user_id
    user_dir.mkdir(parents=True, exist_ok=True)
    qid_path = user_dir / "qid.txt"

    if qid_path.exists():
        qid = int(qid_path.read_text()) + 1
    else:
        qid = 0
    qid_path.write_text(str(qid))
    return qid

def maybe_reuse_model(user_id, threshold=0.85):
    match_uid, sim = find_most_similar_user(user_id, threshold=threshold)
    if match_uid:
        print(f"[model reuse] Reusing {match_uid}'s model for {user_id} (sim={sim:.3f})")
        return match_uid
    return None

def main(user_id="user_1",
         recipe_input_json=None,
         topk=5,
         topn_coarse=20000):
    # 1) I/O init
    maybe_retrain_model(user_id)

    recipes_df = load_recipes()
    ingredient_map = load_ingredient_map()

    # 2) Load user_parents from recipe_input.json (fall back to /data if needed)
    if recipe_input_json is None:
        # prefer project root; then /data
        default_candidates = [
            os.path.join("data", "recipe_input.json"),
            "recipe_input.json",
            "/data/recipe_input.json",
        ]
        recipe_input_json = next((p for p in default_candidates if os.path.exists(p)), default_candidates[-1])

    user_parents = load_user_parents_from_json(recipe_input_json, ingredient_map, conf_th=0.8)

    # 3) Load user profile
    user_profile = ensure_user_profile(user_id)

    # Embedding similarity fallback
    match_uid, sim = find_most_similar_user(user_id, threshold=0.85)
    if match_uid is not None:
        print(f"[main] Using model of similar user '{match_uid}' for '{user_id}' (sim={sim:.3f})")

        src_dir = USER_DATA_DIR / match_uid
        dst_dir = USER_DATA_DIR / user_id
        dst_dir.mkdir(parents=True, exist_ok=True)

        for fname in ["ranker.pkl", "user_features_rank.csv"]:
            src = src_dir / fname
            dst = dst_dir / fname
            if os.path.exists(src) and not os.path.exists(dst):
                shutil.copyfile(src, dst)
                print(f"[embedding] Copied {fname} from {match_uid} to {user_id}")

    # 4) Ensure cold-start features & model
    model_path = ensure_model(user_id)

    # 5) Prepare recipes & coarse rank (Step 2)
    df = prepare_recipes_df(recipes_df)
    recipes_records = df.to_dict(orient="records")

    filtered_records = [r for r in recipes_records if hard_filter(r, user_profile)]
    if not filtered_records:
        print("[main] No recipes after hard dietary filtering.")
        return

    coarse = coarse_rank_candidates(
        recipes=recipes_records,
        user_parents=user_parents,
        user_profile=user_profile,
        top_n=topn_coarse
    )

    if not coarse:
        print("[main] No coarse candidates. Please check user_parents or dataset.")
        return

    # 6) ML reranking (Step 3)
    ml_top = ml_generate_candidates(
        coarse_candidates=coarse,
        user_parents=user_parents,
        user_profile=user_profile,
        model_path=model_path,
        topk=200
    )

    if ml_top is None or len(ml_top) == 0:
        print("[main] No ML candidates returned.")
        return
    
     # 6.5) KMeans Diversification
    candidates_list = ml_top.to_dict(orient="records")
    X_cluster = build_cluster_features(candidates_list)
    diversified = diversify_topk_with_min_clusters(
        ranked_candidates=candidates_list,
        feature_matrix=X_cluster,
        top_k=topk,
        n_clusters=10,
        min_clusters=3
    )

    ml_top = pd.DataFrame(diversified)

    # 7) Pretty print (reuse print_candidates expecting 'match_score')
    ml_top = ml_top.copy()
    if "match_score" not in ml_top.columns and "ml_score" in ml_top.columns:
        ml_top["match_score"] = ml_top["ml_score"]

    print(f"\nFound {len(ml_top)} candidate recipes:\n")
    print_candidates(ml_top, user_parents, topk=topk)

    # 8) Give feedbacks
    qid = get_next_qid(user_id)
    selected_idx = int(input(f"Select a recipe from 1-{topk}: ")) - 1
    selected_row = ml_top.iloc[selected_idx].to_dict()
    collect_user_feedback(user_id, selected_row, user_profile, qid)


def recommend_recipes(detection_payload, user_id, recipes_df, topk=5):
    """
    Unified recommendation entry for the app.
    Handles user profile loading, ingredient mapping, and embedding fallback internally.
    
    """
    # 0) Check if retraining is needed (new feedback, updated features)
    maybe_retrain_model(user_id)
    # 1) Ingredient mapping - use existing high/low confidence fields
    ingredient_map = load_ingredient_map()
    ingredients = detection_payload.get("ingredients", [])

    high_conf = detection_payload.get("high_confidence_ingredients", [])
    low_conf = detection_payload.get("low_confidence_ingredients", [])

    user_parents = []
    for item in ingredients:
        name = item.get("name")
        if not name:
            continue
        parent = ingredient_map.get(name.lower())
        if parent:
            user_parents.append(parent)

    user_parents = sorted(set(user_parents))
    high_conf = sorted(set(high_conf))
    low_conf = sorted(set(low_conf))

    # 2) Load user profile internally
    user_profile = ensure_user_profile(user_id)

    # 3) Embedding fallback
    match_uid, sim = find_most_similar_user(user_id, threshold=0.85)
    if match_uid is not None:
        print(f"[embedding] Using model of similar user '{match_uid}' for '{user_id}' (sim={sim:.3f})")
        src_dir = USER_DATA_DIR / match_uid
        dst_dir = USER_DATA_DIR / user_id
        dst_dir.mkdir(parents=True, exist_ok=True)
        for fname in ["ranker.pkl", "user_features_rank.csv"]:
            src = src_dir / fname
            dst = dst_dir / fname
            if os.path.exists(src) and not os.path.exists(dst):
                shutil.copyfile(src, dst)
                print(f"[embedding] Copied {fname} from {match_uid} to {user_id}")

    # 4) Coldstart / model ensure
    model_path = ensure_model(user_id)

    # 5) Coarse rank
    df = prepare_recipes_df(recipes_df)
    recipes_records = df.to_dict(orient="records")
    filtered_records = [r for r in recipes_records if hard_filter(r, user_profile)]
    if not filtered_records:
        return pd.DataFrame(), user_parents, high_conf, low_conf

    coarse = coarse_rank_candidates(
        recipes=recipes_records,
        user_parents=user_parents,
        user_profile=user_profile,
        top_n=20000
    )
    if not coarse:
        return pd.DataFrame(), user_parents, high_conf, low_conf

    # 6) ML rerank
    ml_top = ml_generate_candidates(
        coarse_candidates=coarse,
        user_parents=user_parents,
        user_profile=user_profile,
        model_path=model_path,
        topk=200
    )
    if ml_top is None or len(ml_top) == 0:
        return pd.DataFrame(), user_parents, high_conf, low_conf

    # 7) KMeans diversification
    candidates_list = ml_top.to_dict(orient="records")
    X_cluster = build_cluster_features(candidates_list)
    diversified = diversify_topk_with_min_clusters(
        ranked_candidates=candidates_list,
        feature_matrix=X_cluster,
        top_k=topk,
        n_clusters=10,
        min_clusters=3
    )

    ml_top = pd.DataFrame(diversified)

    return ml_top, user_parents, high_conf, low_conf


def get_feedback(user_id: str, recipe_row: dict, qid: int = None):
    """
    App-friendly feedback collection function.

    Parameters
    ----------
    user_id : str
        The ID of the user submitting feedback.
    recipe_row : dict
        The recipe information dict (e.g., one row from ml_top.to_dict()).
    qid : int, optional
        The query ID for ranking context. If not provided, defaults to 0 or auto increments.
    """
    # 1) Ensure user profile is loaded internally
    user_profile = ensure_user_profile(user_id)

    # 2) If qid is not provided, generate automatically
    if qid is None:
        try:
            qid = get_next_qid(user_id)
        except Exception:
            qid = 0

    # 3) Delegate to existing collect_user_feedback
    collect_user_feedback(user_id, recipe_row, user_profile, qid)

    print(f"[app] Feedback collected for user '{user_id}', qid={qid}, recipe_id={recipe_row.get('id')}")


if __name__ == "__main__":
    main("user_3")
