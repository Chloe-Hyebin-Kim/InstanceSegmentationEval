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