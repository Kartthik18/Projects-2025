# inference.py
import os, json, csv
import numpy as np
from PIL import Image
from tqdm import tqdm
import supervision as sv
import torch
from rfdetr import RFDETRNano

# Paths
test_images_path = "test_data"
prompts_json_path = "simplified_test_prompts.json"
output_csv_path = "test_predictions.csv"
output_vis_path = "test_visualizations"

os.makedirs(output_vis_path, exist_ok=True)

# Load test prompts
with open(prompts_json_path, "r") as f:
    test_prompts = json.load(f)["images"]

# Load trained model
model = RFDETRNano(
    num_classes=4,
    pretrain_weights="output/checkpoint_best_total.pth"
)

CLASS_NAME_TO_ID = {
    "bulbasaur": 1,
    "charizard": 2,
    "mewtwo": 3,
    "pikachu": 4
}

# --- Helpers ---
def compute_center_of_mass(bbox):
    x_min, y_min, x_max, y_max = bbox
    return [(x_min + x_max) / 2, (y_min + y_max) / 2]

def filter_detections_by_class(detections, target_class_id):
    mask = detections.class_id == target_class_id
    return detections.xyxy[mask] if np.any(mask) else []

# --- Run Inference ---
csv_rows = []
successful_predictions = 0
class_detection_stats = {k: 0 for k in CLASS_NAME_TO_ID.keys()}

for item in tqdm(test_prompts, desc="Processing test images"):
    image_id = item["id"]
    target_class = item["target"][0].lower()
    target_id = CLASS_NAME_TO_ID.get(target_class, None)

    image_file = f"{image_id}.png"
    image_path = os.path.join(test_images_path, image_file)

    if not target_id or not os.path.exists(image_path):
        csv_rows.append([image_file, json.dumps([])])
        continue

    image = Image.open(image_path).convert("RGB")
    detections = model.predict(image, threshold=0.2)

    # Filter detections for this class
    target_bboxes = filter_detections_by_class(detections, target_id)
    if len(target_bboxes) > 0:
        successful_predictions += 1
        class_detection_stats[target_class] += 1
        target_bboxes = np.array(target_bboxes, dtype=np.float32)

        # Save centers in CSV
        targeting_points = [compute_center_of_mass(b) for b in target_bboxes]
        targeting_points = [[float(x), float(y)] for x, y in targeting_points]
        csv_rows.append([image_file, json.dumps(targeting_points)])

        # Save visualization
        filtered_detections = sv.Detections(
            xyxy=target_bboxes,
            class_id=np.array([target_id] * len(target_bboxes)),
            confidence=np.ones(len(target_bboxes))
        )
        annotator = sv.BoxAnnotator()
        annotated_image = np.array(image.copy())
        annotated_image = annotator.annotate(annotated_image, filtered_detections)
        Image.fromarray(annotated_image).save(
            os.path.join(output_vis_path, image_file)
        )
    else:
        csv_rows.append([image_file, json.dumps([])])

# Save CSV
with open(output_csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["image_id", "points"])
    writer.writerows(csv_rows)

print("\n✅ Inference complete")
print(f"CSV saved at {output_csv_path}")
print(f"Annotated images saved at {output_vis_path}")
print(f"Detection stats: {class_detection_stats}")
