readme_text = """\
===========================
Recipe Recommendation System
===========================

This project implements a complete recipe recommendation system, including cold start ranking, ML-based reranking, KMeans-based diversification, and user feedback collection.  
All functions are fully encapsulated and can be easily called from external applications.

-------------------------------------
1. Main Entry Functions for External Use
-------------------------------------

The three main functions for external usage are:

1) recommend_recipes(detection_payload, user_id, recipes_df, topk=5)
   - Input:
     • detection_payload: dict or JSON object containing detected ingredients.
     • user_id: str, unique user identifier.
     • recipes_df: pandas.DataFrame loaded by `load_recipes()`.
     • topk: int, number of final recipes to return (default = 5).
   - Output:
     • ml_top: pandas.DataFrame of top recommended recipes (with ml_score & metadata).
     • user_parents: list of mapped parent ingredients.
     • high_conf: list of high-confidence ingredient matches.
     • low_conf: list of low-confidence or unmapped ingredients.

   Internally, this function performs:
   - Ingredient mapping from detection payload
   - Embedding fallback (copy model/features from similar user)
   - Cold start feature generation if needed
   - Coarse ranking → ML reranking → KMeans diversification
   - Returns the final diversified top-k recommendations.

2) load_recipes()
   - Input: None
   - Output: pandas.DataFrame of all recipes (automatically downloaded from Hugging Face if not present).
   - This function loads the full recipe dataset into memory.  
     If the dataset is not found locally, it will automatically download and cache it under `data/`.

3) get_feedback(user_id, recipe_row, qid=None)
   - Input:
     • user_id: str, unique user identifier.
     • recipe_row: dict, a single recipe row (e.g. one of the top-k recommendations).
     • qid: int, optional query ID. Defaults to auto-generated or 0.
   - Output: None
   - Function:
     • Loads user profile internally
     • Appends the feedback (recipe metadata, user choice) into `user_data/{user_id}/feedback.csv`
     • Does not retrain the model automatically (use `maybe_retrain_model` if needed)

----------------------------------------
2. User Profiles and Pretrained Models
----------------------------------------

The `user_data` folder contains four example users:

- user_0 : Empty profile for testing the system’s ability to bootstrap from zero information.
- user_1 : A user with specific dietary habits.
- user_2 : A user with different dietary preferences.
- user_3 : Similar to user_2, used to test simple embedding-based model reuse.

For each user:
- Cold start features and ML models (`user_features_rank.csv` and `ranker.pkl`) have already been generated.
- You can add new users by creating a new folder under `user_data/` with a profile file `user_profile.json` in the following format:

{
  "user_id": "user_001",
  "num_feedback": 0,
  "diet": {
    "vegetarian_type": "flexible_vegetarian"
  },
  "allergies": ["peanut", "shrimp"],
  "region_preference": ["Asia", "Europe"],
  "nutritional_goals": {
    "calories": { "min": 400, "max": 3000 },
    "protein": { "min": 100, "max": 160 }
  },
  "other_preferences": {
    "preferred_main": ["chicken", "tofu"],
    "disliked_main": ["lamb"],
    "cooking_time_max": 40
  }
}

The cold start process will typically take **15–25 minutes**, depending on your system performance.

----------------------------------------
3. Dataset Download
----------------------------------------

Large recipe and ingredient mapping files are stored on Hugging Face under the account:
  → iris314

These files will be automatically downloaded the first time `load_recipes()` or related functions are called.  
No manual setup is required.

----------------------------------------
4. Feedback Loop & Retraining
----------------------------------------

User feedback is saved in `feedback.csv` files under each user's directory.  
To trigger retraining after feedback collection, call:

from trainmodel import maybe_retrain_model
maybe_retrain_model(user_id)

This checks timestamps between `user_features_rank.csv` and `ranker.pkl` to decide if retraining is needed.

----------------------------------------
5. Cold Start & Embedding Fallback
----------------------------------------

- If a user has no model or features, the system runs a cold start procedure to generate ranking features.
- If a similar user exists (cosine similarity > 0.85), the system copies their model and features to skip retraining.

----------------------------------------
6. Quick Start Example
----------------------------------------

from main import recommend_recipes, load_recipes, get_feedback

# 1. Load dataset
recipes_df = load_recipes()

# 2. Prepare a fake detection payload
payload = {"detected_ingredients": ["chicken", "milk", "flour"]}

# 3. Recommend
top_recipes, user_parents, high_conf, low_conf = recommend_recipes(payload, "user_1", recipes_df, topk=5)

# 4. Feedback
get_feedback("user_1", top_recipes.iloc[0].to_dict())

----------------------------------------
End of README
----------------------------------------
"""

with open("README.txt", "w", encoding="utf-8") as f:
    f.write(readme_text)

"README.txt file created successfully."
