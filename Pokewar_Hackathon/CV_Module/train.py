import os
import shutil
from roboflow import download_dataset
from rfdetr import RFDETRNano

# ----------------------------
# Setup
# ----------------------------
# If running on Colab, set your API key first:
# from google.colab import userdata
# os.environ["ROBOFLOW_API_KEY"] = userdata.get("ROBOFLOW_API_KEY")

print("[INFO] Downloading dataset...")
dataset = download_dataset(
    "https://app.roboflow.com/newplan-iv3ma/object_detection2-qlmrd/1", 
    "coco"
)

# Fix test folder structure
os.makedirs("/content/Object_detection2-1/test", exist_ok=True)
shutil.copy(
    "/content/Pokemon-detect-3/train/_annotations.coco.json",
    "/content/Object_detection2-1/test/_annotations.coco.json"
)

# ----------------------------
# Train RF-DETR
# ----------------------------
print("[INFO] Initializing RF-DETR model...")
model = RFDETRNano()

model.train(
    dataset_dir="/content/Object_detection2-1",
    epochs=12,
    batch_size=4,
    grad_accum_steps=4,
)

print("[INFO] Training complete. Model saved in /content/output")
