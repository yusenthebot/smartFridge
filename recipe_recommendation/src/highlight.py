import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import numpy as np


def print_candidates(candidates, user_parents, topk=10):
    shown = 0
    max_score = candidates['match_score'].max()
    min_score = candidates['match_score'].min()

    for _, row in candidates.head(topk).iterrows():
        scaled_score = 100 * row['match_score'] / (max_score + 1e-9)
        print(f"{row['name']} (score {scaled_score:.1f}%)")

        # ----- Region -----
        region = row.get("region", None)
        if pd.notna(region) and isinstance(region, str) and region.strip() and region.lower() != "unavailable":
            print(f"  region: {region}")

        # ----- Cuisine Attributes -----
        cuisine = row.get("cuisine_attr", None)
        if cuisine is not None and not (isinstance(cuisine, float) and pd.isna(cuisine)):
            # Convert set to list for printing
            if isinstance(cuisine, set):
                cuisine = list(cuisine)
            elif isinstance(cuisine, str):
                cuisine = [cuisine]

            if isinstance(cuisine, list) and len(cuisine) > 0:
                print(f"  cuisine: {', '.join(cuisine)}")

        # ----- Nutrition -----
        print(f"  calories: {row.get('calories', 'N/A')}")

        # ----- Ingredient Marking -----
        def mark_list(lst):
            return [("✅ " + ing) if ing in user_parents else ("❌ " + ing) for ing in lst]

        print(f"  staple:    {mark_list(row.get('staple_parent', []))}")
        print(f"  main:      {mark_list(row.get('main_parent', []))}")
        print(f"  seasoning: {row.get('seasoning_parent', [])}")
        print(f"  other:     {mark_list(row.get('other_parent', []))}")
        print("-" * 40)

        shown += 1

def diversify_topk_with_min_clusters(
    ranked_candidates,
    feature_matrix,
    top_k=5,
    n_clusters=20,
    min_clusters=3,
    random_state=42
):
    """
    Diversify top-k displayed recipes using KMeans clustering.
    Ensures that the final top_k contains at least `min_clusters` distinct clusters.
    """
    if len(ranked_candidates) == 0:
        return []

    n_clusters = min(n_clusters, len(ranked_candidates))
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(feature_matrix)

    # KMeans clustering
    kmeans = KMeans(n_clusters=n_clusters, n_init='auto', random_state=random_state)
    cluster_ids = kmeans.fit_predict(X_scaled)

    # Step 1: pick candidates from distinct clusters until min_clusters reached
    picked = []
    picked_clusters = set()
    for i, c in enumerate(cluster_ids):
        if c not in picked_clusters:
            picked.append(ranked_candidates[i])
            picked_clusters.add(c)
        if len(picked_clusters) >= min_clusters or len(picked) >= top_k:
            break

    # Step 2: fill the rest purely by rank order
    if len(picked) < top_k:
        for i, c in enumerate(cluster_ids):
            if ranked_candidates[i] not in picked:
                picked.append(ranked_candidates[i])
            if len(picked) >= top_k:
                break

    return picked


