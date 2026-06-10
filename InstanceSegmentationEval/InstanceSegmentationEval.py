import json
import torch
from pathlib import Path

import numpy as np
from PIL import Image
from rfdetr import RFDETRSegMedium
from pycocotools import mask as mask_utils

PROJECT_ROOT = Path.cwd()

MODEL_PATH = PROJECT_ROOT / "IS_pretrained_bottom.pt"
DATASET_ROOT = PROJECT_ROOT / "dataset-coco-seg" / "dataset-coco-seg"

VALID_DIR = DATASET_ROOT / "valid"
TEST_DIR = DATASET_ROOT / "test"

VALID_ANN_PATH = VALID_DIR / "_annotations.coco.json"
TEST_ANN_PATH = TEST_DIR / "_annotations.coco.json"

print("\n")
print("-------------------------- PATH --------------------------")

print("PROJECT_ROOT:", PROJECT_ROOT)
print("MODEL_PATH:", MODEL_PATH)
print("MODEL exists:", MODEL_PATH.exists())
print("VALID_ANN exists:", VALID_ANN_PATH.exists())
print("TEST_ANN exists:", TEST_ANN_PATH.exists())

print("\n")
print("-------------------------- DATA SET --------------------------")
with open(VALID_ANN_PATH, "r", encoding="utf-8") as f:
    coco_data = json.load(f)

images = coco_data["images"]
print("valid image count:", len(images))

#with open(VALID_ANN_PATH, "r", encoding="utf-8") as f:
#    valid_data = json.load(f)

#with open(TEST_ANN_PATH, "r", encoding="utf-8") as f:
#    test_data = json.load(f)

#print("valid images:", len(valid_data["images"]))
#print("valid annotations:", len(valid_data["annotations"]))
#print("test images:", len(test_data["images"]))
#print("test annotations:", len(test_data["annotations"]))

print("\n")
print("-------------------------- PRETRAINED MODEL --------------------------")

model = RFDETRSegMedium.from_checkpoint(str(MODEL_PATH))
#ckpt = torch.load(
#    MODEL_PATH,
#    map_location="cpu",
#    weights_only=False
#)

#print("model load success")
#print("checkpoint type:", type(ckpt))

#if isinstance(ckpt, dict):
#    print("checkpoint keys:", ckpt.keys())

#    if "args" in ckpt:
#        print("args:")
#        print(ckpt["args"])

#    if "model" in ckpt:
#        print("model weights type:", type(ckpt["model"]))
#        print("number of weight keys:", len(ckpt["model"]))

print("\n")
print("-------------------------- INFERENCE --------------------------")

predictions = []

for idx, image_info in enumerate(images, start=1):
    image_id = image_info["id"]
    file_name = image_info["file_name"]
    image_path = VALID_DIR / file_name

    print(f"[{idx}/{len(images)}] inference: {file_name}")

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

        # 중요:
        # 모델 class_id는 0부터 background, band, beltloop...
        # COCO category_id는 1부터 background, band, beltloop...
        # 그래서 +1 보정 필요
        coco_category_id = int(class_id) + 1

        predictions.append({
            "image_id": int(image_id),
            "category_id": coco_category_id,
            "segmentation": rle,
            "score": float(score),
        })


        
print("\n")
print("-------------------------- PREDICTION JSON --------------------------")

OUTPUT_JSON_PATH = PROJECT_ROOT / "valid_predictions.json"

with open(OUTPUT_JSON_PATH, "w", encoding="utf-8") as f:
    json.dump(predictions, f)

print()
print("-------------------------- SAVE RESULT --------------------------")
print("saved:", OUTPUT_JSON_PATH)
print("prediction count:", len(predictions))