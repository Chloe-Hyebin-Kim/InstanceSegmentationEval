
import json
from pathlib import Path

import numpy as np
from PIL import Image
from pycocotools import mask as mask_utils
from rfdetr import RFDETRSegMedium


PROJECT_ROOT = Path.cwd()

MODEL_PATH = PROJECT_ROOT / "IS_pretrained_bottom.pt"
DATASET_ROOT = PROJECT_ROOT / "dataset-coco-seg" / "dataset-coco-seg"

TEST_DIR = DATASET_ROOT / "test"
TEST_ANN_PATH = TEST_DIR / "_annotations.coco.json"
TEST_OUTPUT_JSON_PATH = PROJECT_ROOT / "test_predictions.json"


print("-------------------------- TEST PATH --------------------------")
print("MODEL exists:", MODEL_PATH.exists())
print("TEST_ANN exists:", TEST_ANN_PATH.exists())


with open(TEST_ANN_PATH, "r", encoding="utf-8") as f:
    test_coco_data = json.load(f)

test_images = test_coco_data["images"]

print("test image count:", len(test_images))


print()
print("-------------------------- LOAD MODEL --------------------------")
model = RFDETRSegMedium.from_checkpoint(str(MODEL_PATH))
print("model load success")


print()
print("-------------------------- TEST INFERENCE --------------------------")

predictions = []

for idx, image_info in enumerate(test_images, start=1):
    image_id = image_info["id"]
    file_name = image_info["file_name"]
    image_path = TEST_DIR / file_name

    print(f"[{idx}/{len(test_images)}] inference: {file_name}")

    image = Image.open(image_path).convert("RGB")
    detections = model.predict(image, threshold=0.05)

    if detections.mask is None:
        continue

    for mask, class_id, score in zip(
        detections.mask,
        detections.class_id,
        detections.confidence
    ):
        mask = np.asfortranarray(mask.astype(np.uint8))

        rle = mask_utils.encode(mask)
        rle["counts"] = rle["counts"].decode("utf-8")

        predictions.append({
            "image_id": int(image_id),
            "category_id": int(class_id),
            "segmentation": rle,
            "score": float(score),
        })


with open(TEST_OUTPUT_JSON_PATH, "w", encoding="utf-8") as f:
    json.dump(predictions, f)

print()
print("-------------------------- SAVE TEST RESULT --------------------------")
print("saved:", TEST_OUTPUT_JSON_PATH)
print("prediction count:", len(predictions))