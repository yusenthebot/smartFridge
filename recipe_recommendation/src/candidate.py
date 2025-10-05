import pandas as pd
import numpy as np
from .feature import build_features
from .io import load_ingredient_map
import joblib

# Load ingredient map globally to avoid repeated I/O
INGREDIENT_MAP = load_ingredient_map()
PARENTS = INGREDIENT_MAP["parents"]
CHILDREN = INGREDIENT_MAP["children"]

def extract_user_parents(user_ingredients):
    """Map user's ingredients to parent categories"""
    user_parents = set()
    for ing in user_ingredients:
        ing_lower = ing.lower().strip()
        if ing_lower in CHILDREN:
            parent = CHILDREN[ing_lower]["parent"]
            user_parents.add(parent)
        elif ing_lower in PARENTS:
            user_parents.add(ing_lower)
    return user_parents


def hard_filter(recipe, user_profile):
    diet = user_profile.get("diet", {}).get("vegetarian_type", "").lower()
    if diet == "vegan" and not recipe.get("is_vegan_safe", True):
        return False
    if diet in ["vegetarian", "flexible_vegetarian"] and not recipe.get("is_vegetarian_safe", True):
        return False
    return True


COARSE_WEIGHTS = {
    "main_match_ratio": 1.0,
    "staple_match_ratio": 0.3,
    "other_match_ratio": 0.6,
    "low_calorie_penalty": 0.2,
    "preferred_course_overlap": 0.1
}


def coarse_score(features, weights=COARSE_WEIGHTS):
    score = 0.0
    for key, w in weights.items():
        if key in features:
            score += w * features[key]
    return score


def coarse_rank_candidates(recipes, user_parents, user_profile, top_n=30000, weights=COARSE_WEIGHTS):
    """
    Stage 2: Coarse Ranking (NumPy vectorized implementation)
    ---------------------------------------------------------
    Quickly retrieves a subset of candidate recipes by computing 
    ingredient coverage ratios (main / staple / other) between 
    the user's pantry and the recipes using vectorized operations.
    
    This function replaces the original Python loop version 
    for significant speedup during cold start and real-time ranking.
    """
    if not recipes:
        return []

    # === 1. Build parent vocabulary ===
    # Extract all unique parent ingredients across main/staple/other fields.
    all_parents = sorted({
        p for r in recipes 
        for k in ["main_parent", "staple_parent", "other_parent"]
        for p in (r.get(k) or [])
    })
    parent_index = {p: i for i, p in enumerate(all_parents)}
    num_recipes = len(recipes)
    num_parents = len(all_parents)

    # === 2. Construct multi-hot matrices for main, staple, other ===
    # Each row corresponds to a recipe; each column to a parent ingredient.
    main_mat   = np.zeros((num_recipes, num_parents), dtype=np.uint8)
    staple_mat = np.zeros((num_recipes, num_parents), dtype=np.uint8)
    other_mat  = np.zeros((num_recipes, num_parents), dtype=np.uint8)

    for i, r in enumerate(recipes):
        for p in r.get("main_parent", []):
            if p in parent_index:
                main_mat[i, parent_index[p]] = 1
        for p in r.get("staple_parent", []):
            if p in parent_index:
                staple_mat[i, parent_index[p]] = 1
        for p in r.get("other_parent", []):
            if p in parent_index:
                other_mat[i, parent_index[p]] = 1

    # === 3. Encode user pantry as a binary mask ===
    user_mask = np.zeros(num_parents, dtype=np.uint8)
    for p in user_parents:
        if p in parent_index:
            user_mask[parent_index[p]] = 1

    # === 4. Compute ingredient match ratios in batch ===
    # main_ratio = (# of matched main ingredients) / (# of total main ingredients)
    main_total   = main_mat.sum(axis=1)
    staple_total = staple_mat.sum(axis=1)
    other_total  = other_mat.sum(axis=1)

    main_match   = (main_mat @ user_mask)
    staple_match = (staple_mat @ user_mask)
    other_match  = (other_mat @ user_mask)

    main_ratio   = main_match   / np.maximum(main_total, 1)
    staple_ratio = staple_match / np.maximum(staple_total, 1)
    other_ratio  = other_match  / np.maximum(other_total, 1)

    # === 5. Additional coarse ranking signals ===
    # Low-calorie preference & preferred cuisine overlap
    calories = np.array([r.get("calories", 0) for r in recipes], dtype=float)
    calorie_threshold = user_profile.get("calorie_threshold", 9999)
    low_calorie_penalty = (calories <= calorie_threshold).astype(float)

    preferred_course_types = set(user_profile.get("preferred_course_types", []))
    preferred_overlap = np.array([
        len(set(r.get("cuisine_attr", [])) & preferred_course_types)
        for r in recipes
    ], dtype=float)

    # === 6. Compute coarse ranking scores ===
    scores = (
        weights["main_match_ratio"]   * main_ratio +
        weights["staple_match_ratio"] * staple_ratio +
        weights["other_match_ratio"]  * other_ratio +
        weights["low_calorie_penalty"] * low_calorie_penalty +
        weights["preferred_course_overlap"] * preferred_overlap
    )

    # === 7. Select top-N candidates ===
    valid_idx = np.where(scores > 0)[0]
    if valid_idx.size == 0:
        return []

    scores_valid = scores[valid_idx]
    topk = min(top_n, valid_idx.size)

    # Optional dynamic thresholding: keep candidates with score >= 50% of max
    max_score = scores_valid.max()
    keep_mask = scores_valid >= max_score * 0.5
    keep_idx = valid_idx[keep_mask]

    if keep_idx.size == 0:
        return []

    order = np.argsort(scores[keep_idx])[::-1]
    top_idx = keep_idx[order[:topk]]

    # Return the original recipe dicts corresponding to the top candidates
    return [recipes[i] for i in top_idx]


def rule_generate_candidates(df, user_parents, user_profile):
    """
    Step 3: Rule-based reranking of coarse candidates.
    Uses all available features (except vegan/vegetarian filters, which were applied in Step 1)
    to compute a weighted rule-based score for each recipe.
    """

    def score(row):
        # Build recipe_dict for feature extraction
        recipe_dict = {
            "main": row.get("main_parent", set()),
            "staple": row.get("staple_parent", set()),
            "other": row.get("other_parent", set()),
            "seasoning": row.get("seasoning_parent", set()),
            "matched_main": len(row.get("main_parent", set()) & set(user_parents)),
            "matched_staple": len(row.get("staple_parent", set()) & set(user_parents)),
            "matched_other": len(row.get("other_parent", set()) & set(user_parents)),
            "calories": row.get("calories", 0),
            "protein": row.get("protein", 0),
            "fat": row.get("fat", 0),
            "region": row.get("region", ""),
            "cuisine_attr": row.get("cuisine_attr", []),
            "ingredients": row.get("ingredients", []),
            "minutes": row.get("minutes", None),
        }

        # Extract rule features
        feats = build_features(recipe_dict, user_profile)

        # Compute rule-based score
        score = 0.0

        # Ingredient match ratios
        # Main ingredients are weighted most heavily
        score += 2.0 * feats["main_match_ratio"]
        score += 1.0 * feats["staple_match_ratio"]
        score += 1.0 * feats["other_match_ratio"]

        # Nutrition preferences
        # Low calorie preference
        if user_profile.get("low_calorie", False):
            if feats["low_calorie_penalty"]:
                score += 0.5

        # High protein preference
        if user_profile.get("high_protein", False) and feats["protein_ratio"] > 0.25:
            score += 0.3

        # Low fat preference (penalty if fat ratio is too high)
        if user_profile.get("low_fat", False) and feats["fat_ratio"] > 0.35:
            score -= 0.3

        # Region / cuisine / main-type preferences
        score += 0.5 * feats["region_match"]
        score += 0.4 * feats["preferred_course_overlap"]
        score += 0.3 * feats["preferred_main_overlap"]

        # Cooking time preference
        score += 0.3 * feats["within_cooking_time"]

        # Missing ingredients penalty
        # Minor penalty for missing main ingredients (after coarse filtering this is usually small)
        score -= 0.2 * feats["missing_main_count"]

        return max(score, 0.0)

    # Apply scoring over the coarse candidate DataFrame
    df = df.copy()
    df["match_score"] = df.apply(score, axis=1)
    df = df[df["match_score"] > 0]
    if df.empty:
        return df
    df = df.sort_values("match_score", ascending=False).reset_index(drop=True)

    return df


def ml_generate_candidates(coarse_candidates, user_parents, user_profile, model_path, topk=5):
    """
    Step 3: ML-based reranking (directly after Step 2).
    Instead of rule-based prefiltering, use the coarse-ranked candidates (Step 2 output),
    build features in the same format as training, and apply the trained ML model to rerank.
    """

    # Handle empty input
    if coarse_candidates is None or len(coarse_candidates) == 0:
        print("No candidates provided for ML reranking.")
        return pd.DataFrame()

    # If input is a list of dicts (from coarse_rank_candidates), convert to DataFrame
    if isinstance(coarse_candidates, list):
        df = pd.DataFrame(coarse_candidates)
    else:
        df = coarse_candidates.copy()

    if df.empty:
        print("Coarse candidates DataFrame is empty.")
        return df

    # Load trained model
    model = joblib.load(model_path)

    # Build feature DataFrame
    feature_rows = []
    for _, row in df.iterrows():
        recipe_dict = {
            "main": row.get("main_parent", set()),
            "staple": row.get("staple_parent", set()),
            "other": row.get("other_parent", set()),
            "seasoning": row.get("seasoning_parent", set()),
            "matched_main": len(row.get("main_parent", set()) & set(user_parents)),
            "matched_staple": len(row.get("staple_parent", set()) & set(user_parents)),
            "matched_other": len(row.get("other_parent", set()) & set(user_parents)),
            "calories": row.get("calories", 0),
            "protein": row.get("protein", 0),
            "fat": row.get("fat", 0),
            "region": row.get("region", ""),
            "cuisine_attr": row.get("cuisine_attr", []),
            "ingredients": row.get("ingredients", []),
            "minutes": row.get("minutes", None),
        }
        feats = build_features(recipe_dict, user_profile)
        feature_rows.append(feats)

    feature_df = pd.DataFrame(feature_rows)

    # Predict ML scores
    if hasattr(model, "predict_proba"):
        df["ml_score"] = model.predict_proba(feature_df)[:, 1]
    else:
        df["ml_score"] = model.predict(feature_df)

    # Sort by ML score and return top-k candidates
    return df.sort_values("ml_score", ascending=False).head(topk).reset_index(drop=True)



