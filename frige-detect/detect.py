# -*- coding: utf-8 -*-
"""
Detect ingredients using a Roboflow model with preprocessing:
- Resize images to 640x640 if needed.
- Perform detection.
- Classify object sizes via K-Means.
- Generate JSON and annotated image outputs.
"""

import json
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from roboflow import Roboflow
from sklearn.cluster import KMeans
import supervision as sv
import tempfile

def compute_area_ratios(predictions, img_shape):
    """Compute area ratio (bbox area / image area) for each detection."""
    img_area = float(img_shape[0] * img_shape[1])
    ratios = []
    for pred in predictions:
        area = pred["width"] * pred["height"]
        ratios.append(area / img_area)
    return np.array(ratios).reshape(-1, 1)

def cluster_sizes(area_ratios):
    """Cluster area ratios into two groups using K-Means and return size labels."""
    kmeans = KMeans(n_clusters=2, init="k-means++", random_state=0)
    labels = kmeans.fit_predict(area_ratios)
    centroids = kmeans.cluster_centers_.flatten()
    large_cluster = np.argmax(centroids)
    return ["large" if lbl == large_cluster else "small" for lbl in labels]

def detect_and_generate(
    image_path: str,
    api_key: str,
    project_name: str,
    version: int,
    conf_threshold: float = 0.4,
    overlap_threshold: float = 0.3,
    conf_split: float = 0.7,
    output_json: str = "recipe_input.json",
    output_image: str = "annotated_image.jpg"
):
    """
    Resize image if necessary, run detection, classify sizes via K-Means, and
    create both JSON output and annotated image.

    Args:
        image_path (str): Path to the original image.
        api_key (str): Roboflow API key.
        project_name (str): Roboflow project name.
        version (int): Model version.
        conf_threshold (float): Minimum confidence threshold (0–1).
        overlap_threshold (float): NMS overlap threshold (0–1).
        conf_split (float): Threshold for high/low confidence lists.
        output_json (str): Output JSON filename.
        output_image (str): Output annotated image filename.

    Returns:
        dict: Recipe input JSON structure.
    """
    # Load original image
    original_img = cv2.imread(image_path)
    if original_img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    height, width = original_img.shape[:2]

    # Preprocess: resize to 640x640 if needed, and save to a temp file
    if height != 640 or width != 640:
        resized_img = cv2.resize(original_img, (640, 640))
        # create temporary file via mkstemp; close fd to avoid locking
        fd, tmp_path = tempfile.mkstemp(suffix=".jpg")
        os.close(fd)
        cv2.imwrite(tmp_path, resized_img)
        detection_path = tmp_path
        img_for_annotation = resized_img
    else:
        detection_path = image_path
        img_for_annotation = original_img

    # Initialize Roboflow model
    rf = Roboflow(api_key=api_key)
    model = rf.workspace().project(project_name).version(version).model

    # Run prediction
    response = model.predict(
        detection_path,
        confidence=int(conf_threshold * 100),
        overlap=int(overlap_threshold * 100)
    ).json()
    predictions = response["predictions"]

    # Classify sizes using K-Means
    area_ratios = compute_area_ratios(predictions, img_for_annotation.shape)
    size_labels = cluster_sizes(area_ratios)

    # Build JSON structure
    ingredients = []
    high_conf = []
    low_conf = []
    for pred, size_label in zip(predictions, size_labels):
        name = pred["class"]
        conf = pred["confidence"]
        ingredients.append({
            "name": name,
            "quantity": size_label,
            "confidence": round(conf, 2)
        })
        if conf >= conf_split:
            high_conf.append(name)
        else:
            low_conf.append(name)

    recipe_json = {
        "ingredients": ingredients,
        "high_confidence_ingredients": high_conf,
        "low_confidence_ingredients": low_conf
    }

    # Write JSON to file
    with open(output_json, "w", encoding="utf-8") as jf:
        json.dump(recipe_json, jf, indent=4)

    # Annotate image with bounding boxes and confidence labels
    detections = sv.Detections.from_inference(response)
    label_annotator = sv.LabelAnnotator()
    box_annotator = sv.BoxAnnotator()

    labels_for_annotation = [
        f"{pred['class']} ({pred['confidence']:.2f})" for pred in predictions
    ]

    annotated_img = box_annotator.annotate(
        scene=img_for_annotation.copy(),
        detections=detections
    )
    annotated_img = label_annotator.annotate(
        scene=annotated_img,
        detections=detections,
        labels=labels_for_annotation
    )

    cv2.imwrite(output_image, annotated_img)

    # Display annotated image (optional, for notebooks)
    plt.figure(figsize=(8, 8))
    plt.imshow(cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.title("Annotated Image with K-Means Size Labels")
    plt.show()

    # Clean up temporary file
    if height != 640 or width != 640:
        try:
            os.remove(tmp_path)
        except PermissionError:
            # If still locked on Windows, delay deletion or log a warning
            pass

    return recipe_json

# Example call:
result_json = detect_and_generate(
    image_path="demo/t2.jpg",
    api_key="t2nRJrn7ppJIC8RGHdwk",
    project_name="nutrition-object-detection",
    version=1
)
print(json.dumps(result_json, indent=4))
