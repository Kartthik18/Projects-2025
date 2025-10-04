import os
import json
import csv
import numpy as np
from PIL import Image
from tqdm import tqdm
import supervision as sv
from rfdetr import RFDETRNano

# ----------------------------
# Paths
# ----------------------------
test_images_path = "/content/test_data"
prompts_json_path = "/content/simplified_test_prompts.json"
output_csv_path = "/content/test_predictions.csv"
output_vis_path = "/content/test_visualization"

os.makedirs(output_vis_path, exist_ok=True)

# ----------------------------
# Load prompts
# ----------------------------
with open(prompts_json_path, "r") as f:
    test_prompts = json.load(f)["images"]

# ----------------------------
# Load trained model
# ----------------------------
print("[INFO] Loading trained RF-DETR...")
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

# ----------------------------
# Helpers
# ----------------------------
def compute_center_of_mass(bbox):
    x_min, y_min, x_max, y_max = bbox
    return [(x_min + x_max) / 2, (y_min + y_max) / 2]

def filter_detections_by_class(detections, target_class_id):
    mask = detections.class_id == target_class_id
    return detections.xyxy[mask] if np.any(mask) else []

# ----------------------------
# Run inference
# ----------------------------
csv_rows = []
class_detection_stats = {name: 0 for name in CLASS_NAME_TO_ID.keys()}
success_count = 0

print(f"[INFO] Running inference on {len(test_prompts)} images...")

for item in tqdm(test_prompts):
    image_id = item["id"]
    target_class = item["target"][0].lower()
    img_path = os.path.join(test_images_path, f"{image_id}.png")

    if target_class not in CLASS_NAME_TO_ID:
        csv_rows.append([f"{image_id}.png", json.dumps([])])
        continue

    if not os.path.exists(img_path):
        csv_rows.append([f"{image_id}.png", json.dumps([])])
        continue

    detections = model.predict(Image.open(img_path).convert('RGB'), threshold=0.2)
    target_bboxes = filter_detections_by_class(detections, CLASS_NAME_TO_ID[target_class])

    if len(target_bboxes) > 0:
        success_count += 1
        class_detection_stats[target_class] += 1
        centers = [compute_center_of_mass(b) for b in target_bboxes]
        csv_rows.append([f"{image_id}.png", json.dumps(centers)])

        # Visualization
        filtered = sv.Detections(
            xyxy=np.array(target_bboxes, dtype=np.float32),
            class_id=np.array([CLASS_NAME_TO_ID[target_class]] * len(target_bboxes)),
            confidence=np.ones(len(target_bboxes))
        )
        annotator = sv.BoxAnnotator()
        annotated = annotator.annotate(np.array(Image.open(img_path)), filtered)
        Image.fromarray(annotated).save(os.path.join(output_vis_path, f"{image_id}.png"))
    else:
        csv_rows.append([f"{image_id}.png", json.dumps([])])

# Save CSV
with open(output_csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["image_id", "points"])
    writer.writerows(csv_rows)

print(f"[INFO] Done! {success_count}/{len(test_prompts)} detections succeeded.")
print(f"[INFO] Results saved to {output_csv_path}, visualizations in {output_vis_path}")
