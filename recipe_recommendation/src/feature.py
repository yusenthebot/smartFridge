import json
from .io import load_ingredient_map
import numpy as np

# Load ingredient map globally to avoid repeated I/O
INGREDIENT_MAP = load_ingredient_map()
PARENTS = INGREDIENT_MAP["parents"]
CHILDREN = INGREDIENT_MAP["children"]


def is_recipe_vegetarian_safe(ingredients: list[str], veg_type: str) -> bool:
    """
    Check if the recipe is safe for a given dietary type.
    Supported veg_type: "vegan", "vegetarian", "flexible_vegetarian", "" (none).
    """
    for ing in ingredients:
        ing_lower = ing.strip().lower()
        if ing_lower in CHILDREN:
            info = CHILDREN[ing_lower]
        elif ing_lower in PARENTS:
            info = PARENTS[ing_lower]
        else:
            # If the ingredient is not found in the map, treat it as safe by default.
            continue

        if veg_type == "vegan" and not info.get("vegan_safe", True):
            return False
        if veg_type == "vegetarian" and not info.get("vegetarian_safe", True):
            return False
        if veg_type == "flexible_vegetarian":
            # Flexible vegetarians allow most ingredients except explicit meat.
            # Here, we can use vegetarian_safe as a proxy for flexibility.
            if not info.get("vegetarian_safe", True):
                return False
    return True


def build_features(recipe: dict, user_profile: dict) -> dict:
    """
    Build a feature dictionary for ML ranker and rule-based scoring.
    All features are numeric scalars or counts.
    """
    features = {}

    # Ingredient matching ratios
    total_main = len(recipe.get("main", []))
    total_other = len(recipe.get("other", []))
    total_staple = len(recipe.get("staple", []))

    features["main_match_ratio"] = recipe.get("matched_main", 0) / max(total_main, 1)
    features["other_match_ratio"] = recipe.get("matched_other", 0) / max(total_other, 1)
    features["staple_match_ratio"] = recipe.get("matched_staple", 0) / max(total_staple, 1)

    features["missing_main_count"] = total_main - recipe.get("matched_main", 0)
    features["missing_other_count"] = total_other - recipe.get("matched_other", 0)
    features["missing_staple_count"] = total_staple - recipe.get("matched_staple", 0)

    # Nutrition information
    calories = recipe.get("calories", 0)
    protein = recipe.get("protein", 0)
    fat = recipe.get("fat", 0)
    features["calories"] = calories
    features["protein"] = protein
    features["fat"] = fat
    features["protein_ratio"] = protein / max(calories, 1)
    features["fat_ratio"] = fat / max(calories, 1)

    # Regional preference
    recipe_region = recipe.get("region", "")
    if isinstance(recipe_region, set):
        features["region_match"] = int(any(
            r in user_profile.get("preferred_regions", []) for r in recipe_region
        ))
    else:
        features["region_match"] = int(
            recipe_region in user_profile.get("preferred_regions", [])
        )

    # Diet constraints
    ingredients_all = recipe.get("ingredients", [])

    # Vegan-safe check (absolute, independent of user)
    features["is_vegan_safe"] = int(is_recipe_vegetarian_safe(ingredients_all, "vegan"))

    # Vegetarian-safe check (absolute, independent of user)
    features["is_vegetarian_safe_absolute"] = int(
        is_recipe_vegetarian_safe(ingredients_all, "vegetarian")
    )

    # Flexible vegetarian-safe check (absolute, independent of user)
    features["is_flexible_safe_absolute"] = int(
        is_recipe_vegetarian_safe(ingredients_all, "flexible_vegetarian")
    )

    # User diet safety (depends on user_profile)
    veg_type = (user_profile.get("diet", {}).get("vegetarian_type", "") or "").lower()
    features["is_user_diet_safe"] = int(is_recipe_vegetarian_safe(ingredients_all, veg_type))

    # Calorie preference
    calorie_threshold = user_profile.get("calorie_threshold", 9999)
    features["low_calorie_penalty"] = int(calories <= calorie_threshold)

    # Main ingredient preference
    preferred_main = set(user_profile.get("other_preferences", {}).get("preferred_main", []))
    recipe_main = set(recipe.get("main", []))
    features["preferred_main_overlap"] = len(recipe_main & preferred_main)

    # Course type preference
    # e.g. user may prefer 'main-dish' or 'desserts'
    recipe_types = set(recipe.get("cuisine_attr", []))
    preferred_types = set(user_profile.get("preferred_course_types", []))
    features["preferred_course_overlap"] = len(recipe_types & preferred_types)

    # Cooking time preference
    cooking_time_max = user_profile.get("other_preferences", {}).get("cooking_time_max", None)
    if cooking_time_max:
        features["within_cooking_time"] = int(recipe.get("minutes", 9999) <= cooking_time_max)
    else:
        features["within_cooking_time"] = 1

    return features

def build_cluster_features(candidates):
    """
    Build simple ingredient + cuisine based feature vectors for KMeans clustering.
    This is separate from model training features.
    
    Args:
        candidates (list[dict]): list of recipe dicts.
    
    Returns:
        np.ndarray: feature matrix (num_candidates, num_features)
    """
    # 1. Collect vocabulary for ingredients and cuisine
    all_main = set()
    all_staple = set()
    all_other = set()
    all_cuisine = set()

    for r in candidates:
        all_main.update(r.get("main_parent", []) or [])
        all_staple.update(r.get("staple_parent", []) or [])
        all_other.update(r.get("other_parent", []) or [])
        all_cuisine.update(r.get("cuisine_attr", []) or [])

    main_vocab = sorted(all_main)
    staple_vocab = sorted(all_staple)
    other_vocab = sorted(all_other)
    cuisine_vocab = sorted(all_cuisine)

    # 2. Build index map
    main_idx = {p: i for i, p in enumerate(main_vocab)}
    staple_idx = {p: i + len(main_vocab) for i, p in enumerate(staple_vocab)}
    other_idx = {p: i + len(main_vocab) + len(staple_vocab) for i, p in enumerate(other_vocab)}
    cuisine_idx = {p: i + len(main_vocab) + len(staple_vocab) + len(other_vocab) 
                   for i, p in enumerate(cuisine_vocab)}

    dim = len(main_vocab) + len(staple_vocab) + len(other_vocab) + len(cuisine_vocab)
    X = np.zeros((len(candidates), dim), dtype=np.uint8)

    # 3. Fill feature matrix
    for i, r in enumerate(candidates):
        for p in r.get("main_parent", []) or []:
            if p in main_idx:
                X[i, main_idx[p]] = 1
        for p in r.get("staple_parent", []) or []:
            if p in staple_idx:
                X[i, staple_idx[p]] = 1
        for p in r.get("other_parent", []) or []:
            if p in other_idx:
                X[i, other_idx[p]] = 1
        for p in r.get("cuisine_attr", []) or []:
            if p in cuisine_idx:
                X[i, cuisine_idx[p]] = 1

    return X
