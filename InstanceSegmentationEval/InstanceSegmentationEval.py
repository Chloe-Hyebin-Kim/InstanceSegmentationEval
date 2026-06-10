import json
import torch
from pathlib import Path

print("-------------------------- DATA SET --------------------------")

dataset_root = Path("dataset-coco-seg/dataset-coco-seg")

valid_ann_path = dataset_root / "valid" / "_annotations.coco.json"
test_ann_path = dataset_root / "test" / "_annotations.coco.json"

print("valid annotation:", valid_ann_path.exists())
print("test annotation:", test_ann_path.exists())
print("model:", Path("IS_pretrained_bottom.pt").exists())

with open(valid_ann_path, "r", encoding="utf-8") as f:
    valid_data = json.load(f)

with open(test_ann_path, "r", encoding="utf-8") as f:
    test_data = json.load(f)

print("valid images:", len(valid_data["images"]))
print("valid annotations:", len(valid_data["annotations"]))
print("test images:", len(test_data["images"]))
print("test annotations:", len(test_data["annotations"]))



print("valid annotations:", len(valid_data["annotations"]))


print("\n\n")
print("-------------------------- PRETRAINED MODEL --------------------------")

model_path = Path("IS_pretrained_bottom.pt")

print("model exists:", model_path.exists())

ckpt = torch.load(
    model_path,
    map_location="cpu",
    weights_only=False
)

print("model load success")
print("checkpoint type:", type(ckpt))

if isinstance(ckpt, dict):
    print("checkpoint keys:", ckpt.keys())

    if "args" in ckpt:
        print("args:")
        print(ckpt["args"])

    if "model" in ckpt:
        print("model weights type:", type(ckpt["model"]))
        print("number of weight keys:", len(ckpt["model"]))