import json
from pathlib import Path

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