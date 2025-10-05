import os
import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def profile_to_embedding(profile):
    """
    Convert a normalized user profile into a fixed-length numeric embedding.
    Embedding structure:
    [diet (3)] + [allergies (6)] + [region (6)] +
    [nutritional goals (4)] + [preferred_main (8)] + [cooking_time (1)]
    Total dim ≈ 28
    """
    vecs = []

    # 1. Diet (one-hot)
    diet_types = ["vegetarian", "flexible", "non_vegetarian"]
    diet_vec = np.zeros(len(diet_types))
    diet_value = profile.get("diet", {}).get("vegetarian_type", "flexible")
    if diet_value in diet_types:
        diet_vec[diet_types.index(diet_value)] = 1
    vecs.append(diet_vec)

    # 2. Allergies (multi-hot)
    allergy_vocab = ["milk", "gluten", "peanut", "shrimp", "egg", "soy"]
    allergies = set(profile.get("allergies", []))
    allergy_vec = np.array([1 if a in allergies else 0 for a in allergy_vocab])
    vecs.append(allergy_vec)

    # 3. Region preferences (multi-hot)
    region_vocab = ["North America", "Latin America", "Europe", "Asia", "Middle East", "Africa"]
    regions = set(profile.get("region_preference", []))
    region_vec = np.array([1 if r in regions else 0 for r in region_vocab])
    vecs.append(region_vec)

    # 4. Nutritional goals (normalized)
    ng = profile.get("nutritional_goals", {})
    cal = ng.get("calories", {})
    pro = ng.get("protein", {})

    cal_min = cal.get("min", 0) / 4000
    cal_max = min(cal.get("max", 9999), 4000) / 4000
    pro_min = pro.get("min", 0) / 300
    pro_max = min(pro.get("max", 999), 300) / 300

    vecs.append(np.array([cal_min, cal_max, pro_min, pro_max]))

    # 5. Preferred main ingredients (multi-hot)
    main_vocab = ["chicken", "tofu", "beef", "salmon", "eggs", "pork", "beans", "mushroom"]
    mains = set(profile.get("other_preferences", {}).get("preferred_main", []))
    main_vec = np.array([1 if m in mains else 0 for m in main_vocab])
    vecs.append(main_vec)

    # 6. Cooking time max (normalized to [0,1], assume 120 min upper bound)
    t = profile.get("other_preferences", {}).get("cooking_time_max")
    t_vec = np.array([min(t / 120, 1)]) if t is not None else np.array([0])
    vecs.append(t_vec)

    return np.concatenate(vecs)


def profile_similarity(profile_a, profile_b):
    """Compute cosine similarity between two user profiles."""
    emb_a = profile_to_embedding(profile_a).reshape(1, -1)
    emb_b = profile_to_embedding(profile_b).reshape(1, -1)
    return cosine_similarity(emb_a, emb_b)[0, 0]

def find_most_similar_user(target_user_id, user_data_dir="user_data", threshold=0.85):
    """
    Find the most similar existing user based on profile embeddings.
    Returns (best_match_user_id, similarity_score) or (None, -1) if no match.
    """
    target_profile_path = os.path.join(user_data_dir, target_user_id, "user_profile.json")
    if not os.path.exists(target_profile_path):
        raise FileNotFoundError(f"[embedding] No profile found for user {target_user_id}")

    with open(target_profile_path, "r", encoding="utf-8") as f:
        target_profile = json.load(f)
    target_emb = profile_to_embedding(target_profile).reshape(1, -1)

    best_match, best_score = None, -1

    for uid in os.listdir(user_data_dir):
        if uid == target_user_id:
            continue
        profile_path = os.path.join(user_data_dir, uid, "user_profile.json")
        if not os.path.exists(profile_path):
            continue
        with open(profile_path, "r", encoding="utf-8") as f:
            other_profile = json.load(f)
        other_emb = profile_to_embedding(other_profile).reshape(1, -1)
        sim = cosine_similarity(target_emb, other_emb)[0, 0]
        if sim > best_score:
            best_match, best_score = uid, sim

    if best_match and best_score >= threshold:
        print(f"[embedding] Found similar user: {best_match} (similarity={best_score:.3f})")
        return best_match, best_score

    return None, -1
