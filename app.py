"""Gradio application for the smart fridge detector + recipe recommendation pipeline."""

import json
import tempfile
from pathlib import Path
from typing import List, Tuple, Dict, Any

import cv2
import gradio as gr
import numpy as np
from PIL import Image

from frige_detect.detect import (
    detect_and_generate,
    load_roboflow_credentials,
    RoboflowCredentials,
)
from recipe_recommendation.main import (
    load_recipes,
    recommend_recipes,
    save_user_profile,
    get_feedback,
    USER_DATA_DIR,
)

# ---------------------------------------------------------------------------
# Global resources
# ---------------------------------------------------------------------------
CREDENTIALS_PATH = Path("frige_detect/roboflow_credentials.txt")
ROBOFLOW_CREDENTIALS: RoboflowCredentials = load_roboflow_credentials(str(CREDENTIALS_PATH))
RECIPES_DF = load_recipes()
EXAMPLE_IMAGES = [
    "frige_detect/demo/t1.jpg",
    "frige_detect/demo/t2.jpg",
    "frige_detect/demo/t3.jpg",
]


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------
def parse_csv_list(text: str) -> List[str]:
    if not text:
        return []
    parts = [item.strip() for item in text.split(",") if item.strip()]
    return parts


def ensure_numpy_image(image: Any) -> np.ndarray:
    """Convert incoming image (PIL or numpy) to RGB numpy array."""
    if image is None:
        raise ValueError("Please upload a fridge photo before running detection.")
    if isinstance(image, np.ndarray):
        # Assume already RGB
        return image
    if isinstance(image, Image.Image):
        return np.array(image.convert("RGB"))
    raise ValueError("Unsupported image format provided.")


def write_temp_image(image: np.ndarray) -> str:
    """Write numpy image to a temporary file and return the path."""
    temp_dir = Path(tempfile.mkdtemp(prefix="fridge_upload_"))
    temp_path = temp_dir / "upload.jpg"
    bgr_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(temp_path), bgr_image)
    return str(temp_path)


def build_user_profile(
    user_id: str,
    vegetarian_type: str,
    allergies: str,
    regions: str,
    calorie_range: Tuple[float, float],
    protein_range: Tuple[float, float],
    preferred_main: str,
    disliked_main: str,
    cooking_time: float,
) -> Dict[str, Any]:
    user_id = user_id.strip()
    if not user_id:
        raise ValueError("User ID cannot be empty.")

    profile_dir = USER_DATA_DIR / user_id
    profile_path = profile_dir / "user_profile.json"
    if profile_path.exists():
        existing = json.loads(profile_path.read_text(encoding="utf-8"))
        num_feedback = existing.get("num_feedback", 0)
    else:
        num_feedback = 0

    profile = {
        "user_id": user_id,
        "num_feedback": num_feedback,
        "diet": {"vegetarian_type": vegetarian_type},
        "allergies": parse_csv_list(allergies),
        "region_preference": parse_csv_list(regions),
        "nutritional_goals": {
            "calories": {"min": int(calorie_range[0]), "max": int(calorie_range[1])},
            "protein": {"min": int(protein_range[0]), "max": int(protein_range[1])},
        },
        "other_preferences": {
            "preferred_main": parse_csv_list(preferred_main),
            "disliked_main": parse_csv_list(disliked_main),
            "cooking_time_max": int(cooking_time) if cooking_time else None,
        },
    }

    save_user_profile(user_id, profile)
    return profile


def summarize_ingredients(
    user_parents: List[str],
    high_conf: List[str],
    low_conf: List[str],
) -> str:
    lines = ["### Ingredient Mapping"]
    if user_parents:
        lines.append("- **Mapped parent ingredients:** " + ", ".join(sorted(user_parents)))
    else:
        lines.append("- **Mapped parent ingredients:** none")
    if high_conf:
        lines.append("- **High confidence detections:** " + ", ".join(sorted(high_conf)))
    if low_conf:
        lines.append("- **Low confidence detections:** " + ", ".join(sorted(set(low_conf))))
    return "\n".join(lines)


def _ensure_iterable(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, set):
        return sorted(value)
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        return [value]
    return list(value)


def render_recommendations(df) -> Tuple[str, List[Dict[str, Any]]]:
    if df is None or df.empty:
        return "No recipes matched the current constraints.", []

    lines = ["### Recommended Recipes"]
    feedback_rows: List[Dict[str, Any]] = []

    for idx, row in df.head(5).iterrows():
        match_score = row.get("match_score") or row.get("ml_score", 0)
        scaled = match_score * 100 if match_score is not None else 0
        name = row.get("name", f"Recipe {idx+1}")
        lines.append(f"{idx + 1}. **{name}** — score {scaled:.1f}%")

        region = row.get("region")
        if region and not (isinstance(region, float) and np.isnan(region)):
            if isinstance(region, (set, list)):
                region_str = ", ".join(sorted(region))
            else:
                region_str = str(region)
            lines.append(f"   - Region: {region_str}")

        cuisine = row.get("cuisine_attr")
        cuisine_items = _ensure_iterable(cuisine)
        if cuisine_items:
            lines.append(f"   - Cuisine: {', '.join(cuisine_items)}")

        calories = row.get("calories")
        protein = row.get("protein")
        if calories is not None:
            lines.append(f"   - Calories: {calories}")
        if protein is not None:
            lines.append(f"   - Protein: {protein}")

        for key in ["main_parent", "staple_parent", "other_parent"]:
            parents = _ensure_iterable(row.get(key))
            if parents:
                pretty_key = key.replace("_", " ").title()
                lines.append(f"   - {pretty_key}: {', '.join(parents)}")

        ingredients = row.get("ingredients")
        if ingredients:
            if isinstance(ingredients, str):
                ingredients_list = parse_csv_list(ingredients)
            else:
                ingredients_list = list(ingredients)
            if ingredients_list:
                lines.append(f"   - Ingredients: {', '.join(ingredients_list[:10])}")
        lines.append("")

        feedback_row = row.to_dict()
        for key in ["main_parent", "staple_parent", "other_parent", "seasoning_parent", "cuisine_attr", "ingredients"]:
            value = feedback_row.get(key)
            if isinstance(value, list):
                feedback_row[key] = set(value)
            elif isinstance(value, str):
                feedback_row[key] = set(parse_csv_list(value))
        feedback_rows.append(feedback_row)

    return "\n".join(lines).strip(), feedback_rows


def run_pipeline(
    image,
    user_id,
    vegetarian_type,
    allergies,
    regions,
    calorie_range,
    protein_range,
    preferred_main,
    disliked_main,
    cooking_time,
):
    try:
        rgb_image = ensure_numpy_image(image)
        upload_path = write_temp_image(rgb_image)
        temp_dir = Path(tempfile.mkdtemp(prefix="fridge_outputs_"))
        output_json = temp_dir / "recipe_input.json"
        output_image = temp_dir / "annotated_image.jpg"

        detection_result = detect_and_generate(
            image_path=upload_path,
            credentials=ROBOFLOW_CREDENTIALS,
            conf_threshold=0.4,
            overlap_threshold=0.3,
            conf_split=0.7,
            output_json=str(output_json),
            output_image=str(output_image),
        )
        Path(upload_path).unlink(missing_ok=True)

        profile = build_user_profile(
            user_id,
            vegetarian_type,
            allergies,
            regions,
            calorie_range,
            protein_range,
            preferred_main,
            disliked_main,
            cooking_time,
        )

        detection_payload = detection_result["recipe_json"]
        ml_top, user_parents, high_conf, low_conf = recommend_recipes(
            detection_payload,
            user_id,
            RECIPES_DF,
            topk=5,
        )

        ingredient_summary = summarize_ingredients(user_parents, high_conf, low_conf)
        recommendation_md, feedback_rows = render_recommendations(ml_top)

        dropdown_choices = [
            f"{idx + 1}. {row.get('name', 'Recipe')}" for idx, row in enumerate(feedback_rows)
        ]

        status = "" if feedback_rows else "No recipes available for feedback yet."

        return (
            str(output_image),
            detection_payload,
            ingredient_summary,
            recommendation_md,
            gr.Dropdown.update(choices=dropdown_choices, value=None),
            feedback_rows,
            status,
        )
    except Exception as exc:
        return (
            None,
            None,
            "",
            "",
            gr.Dropdown.update(choices=[], value=None),
            [],
            f"❗️ Error: {exc}",
        )


def record_feedback(selected_recipe: str, user_id: str, feedback_rows: List[Dict[str, Any]]):
    if not selected_recipe:
        return "Please select a recipe before submitting feedback."
    if not user_id:
        return "Please provide a valid user ID."
    if not feedback_rows:
        return "No recommendation data available. Run the pipeline first."

    try:
        index = int(selected_recipe.split(".")[0]) - 1
    except (ValueError, IndexError):
        return "Unable to parse the selected recipe."

    if index < 0 or index >= len(feedback_rows):
        return "Selected recipe is out of range."

    recipe_row = feedback_rows[index]
    get_feedback(user_id, recipe_row)

    profile_path = USER_DATA_DIR / user_id / "user_profile.json"
    if profile_path.exists():
        data = json.loads(profile_path.read_text(encoding="utf-8"))
        data["num_feedback"] = data.get("num_feedback", 0) + 1
        save_user_profile(user_id, data)

    return f"Feedback recorded for {recipe_row.get('name', 'selected recipe')}!"


def list_existing_users() -> List[str]:
    if not USER_DATA_DIR.exists():
        return []
    return sorted([p.name for p in USER_DATA_DIR.iterdir() if p.is_dir()])


# ---------------------------------------------------------------------------
# Gradio UI definition
# ---------------------------------------------------------------------------
with gr.Blocks(title="Smart Fridge Recipe Assistant", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # 🧊 Smart Fridge Recipe Assistant
        Upload a fridge photo, tweak your dietary preferences, and receive tailored recipe ideas.
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(
                label="Fridge photo",
                type="pil",
                height=350,
            )
            gr.Examples(
                examples=[[path] for path in EXAMPLE_IMAGES],
                inputs=image_input,
                label="Quick start examples",
            )
            detection_json = gr.JSON(label="Detection payload")
            annotated_output = gr.Image(label="Annotated detection", height=350)

        with gr.Column(scale=1):
            existing_users = list_existing_users()
            user_id_box = gr.Textbox(
                label="User ID",
                value=existing_users[0] if existing_users else "user_1",
                placeholder="e.g. user_1",
            )
            vegetarian_radio = gr.Radio(
                [
                    "flexible",
                    "flexible_vegetarian",
                    "ovo_vegetarian",
                    "lacto_vegetarian",
                    "vegan",
                    "non_vegetarian",
                ],
                label="Vegetarian preference",
                value="flexible",
            )
            allergies_box = gr.Textbox(
                label="Allergies (comma separated)",
                placeholder="peanut, shrimp",
            )
            regions_box = gr.Textbox(
                label="Preferred regions (comma separated)",
                placeholder="Asia, Europe",
            )
            calorie_slider = gr.RangeSlider(
                minimum=0,
                maximum=4000,
                value=(400, 2000),
                label="Calorie range",
                step=50,
            )
            protein_slider = gr.RangeSlider(
                minimum=0,
                maximum=250,
                value=(50, 160),
                label="Protein range",
                step=5,
            )
            preferred_box = gr.Textbox(
                label="Preferred main ingredients",
                placeholder="chicken, tofu",
            )
            disliked_box = gr.Textbox(
                label="Disliked main ingredients",
                placeholder="lamb",
            )
            cooking_slider = gr.Slider(
                minimum=0,
                maximum=180,
                value=45,
                step=5,
                label="Max cooking time (minutes)",
            )
            run_button = gr.Button("Analyze fridge & recommend recipes", variant="primary")
            ingredient_md = gr.Markdown()
            recommendation_md = gr.Markdown()
            feedback_dropdown = gr.Dropdown(label="Select a recipe for positive feedback", choices=[])
            feedback_button = gr.Button("Save feedback")
            feedback_status = gr.Markdown()
            feedback_state = gr.State([])

    run_button.click(
        fn=run_pipeline,
        inputs=[
            image_input,
            user_id_box,
            vegetarian_radio,
            allergies_box,
            regions_box,
            calorie_slider,
            protein_slider,
            preferred_box,
            disliked_box,
            cooking_slider,
        ],
        outputs=[
            annotated_output,
            detection_json,
            ingredient_md,
            recommendation_md,
            feedback_dropdown,
            feedback_state,
            feedback_status,
        ],
    )

    feedback_button.click(
        fn=record_feedback,
        inputs=[feedback_dropdown, user_id_box, feedback_state],
        outputs=feedback_status,
    )

if __name__ == "__main__":
    demo.launch()
