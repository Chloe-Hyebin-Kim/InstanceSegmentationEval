
import json
import shutil
from pathlib import Path

import cv2
import numpy as np
import torch
from pycocotools import mask as mask_utils
from ultralytics import YOLO


# ============================================================
# 1. PATH 설정
# ============================================================

PROJECT_ROOT = Path(r"D:\git\InstanceSegmentationEval\InstanceSegmentationEval")

DATASET_ROOT = PROJECT_ROOT / "dataset-coco-seg"

TRAIN_DIR = DATASET_ROOT / "train"
VALID_DIR = DATASET_ROOT / "valid"
TEST_DIR = DATASET_ROOT / "test"

TRAIN_ANN_PATH = TRAIN_DIR / "_annotations.coco.json"
VALID_ANN_PATH = VALID_DIR / "_annotations.coco.json"
TEST_ANN_PATH = TEST_DIR / "_annotations.coco.json"

YOLO_DATASET_ROOT = PROJECT_ROOT / "dataset-yolo-seg-assignment02"
YOLO_IMAGES_ROOT = YOLO_DATASET_ROOT / "images"
YOLO_LABELS_ROOT = YOLO_DATASET_ROOT / "labels"
YOLO_DATA_YAML_PATH = YOLO_DATASET_ROOT / "data.yaml"

RUNS_ROOT = PROJECT_ROOT / "runs_assignment02"
RUN_NAME = "yolo_seg_assignment02"

# YOLO11이 안 되면 yolov8s-seg.pt로 자동 fallback
MODEL_CANDIDATES = [
    "yolo11s-seg.pt",
    "yolov8s-seg.pt",
]

EPOCHS = 100
IMGSZ = 640
BATCH = 4
PATIENCE = 30

DEVICE = 0 if torch.cuda.is_available() else "cpu"


# ============================================================
# 2. 경로 확인
# ============================================================

def check_paths():
    print()
    print("-------------------------- PATH CHECK --------------------------")
    print("PROJECT_ROOT:", PROJECT_ROOT)
    print("DATASET_ROOT:", DATASET_ROOT)
    print("TRAIN_ANN:", TRAIN_ANN_PATH)
    print("VALID_ANN:", VALID_ANN_PATH)
    print("TEST_ANN:", TEST_ANN_PATH)
    print("DEVICE:", DEVICE)
    print()

    required_paths = [
        TRAIN_DIR,
        VALID_DIR,
        TEST_DIR,
        TRAIN_ANN_PATH,
        VALID_ANN_PATH,
        TEST_ANN_PATH,
    ]

    for path in required_paths:
        print(path.exists(), "->", path)

        if not path.exists():
            raise FileNotFoundError(path)


# ============================================================
# 3. COCO segmentation → YOLO segmentation 변환
# ============================================================

def make_dirs():
    for split in ["train", "val", "test"]:
        (YOLO_IMAGES_ROOT / split).mkdir(parents=True, exist_ok=True)
        (YOLO_LABELS_ROOT / split).mkdir(parents=True, exist_ok=True)


def normalize_polygon(polygon, width, height):
    if len(polygon) < 6:
        return None

    points = []

    for i in range(0, len(polygon), 2):
        x = float(polygon[i])
        y = float(polygon[i + 1])

        x = max(0.0, min(x, width - 1))
        y = max(0.0, min(y, height - 1))

        x_norm = x / width
        y_norm = y / height

        points.extend([x_norm, y_norm])

    if len(points) < 6:
        return None

    return points


def rle_to_largest_polygon(segmentation, height, width):
    if isinstance(segmentation.get("counts"), list):
        rle = mask_utils.frPyObjects(segmentation, height, width)
    else:
        rle = segmentation

        if isinstance(rle["counts"], str):
            rle = {
                "size": rle["size"],
                "counts": rle["counts"].encode("utf-8"),
            }

    mask = mask_utils.decode(rle)

    if mask.ndim == 3:
        mask = np.any(mask, axis=2).astype(np.uint8)
    else:
        mask = mask.astype(np.uint8)

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    if len(contours) == 0:
        return None

    largest_contour = max(contours, key=cv2.contourArea)

    if len(largest_contour) < 3:
        return None

    polygon = largest_contour.reshape(-1, 2).astype(float).flatten().tolist()

    if len(polygon) < 6:
        return None

    return polygon


def get_annotation_polygon(annotation, image_width, image_height):
    segmentation = annotation.get("segmentation", None)

    if segmentation is None:
        return None

    # COCO polygon format
    if isinstance(segmentation, list):
        valid_polygons = []

        for polygon in segmentation:
            if polygon is None:
                continue

            if len(polygon) >= 6:
                valid_polygons.append(polygon)

        if len(valid_polygons) == 0:
            return None

        # 가장 긴 polygon 사용
        return max(valid_polygons, key=len)

    # COCO RLE format
    if isinstance(segmentation, dict):
        return rle_to_largest_polygon(
            segmentation=segmentation,
            height=image_height,
            width=image_width,
        )

    return None


def convert_one_split(coco_split_name, yolo_split_name, image_dir, annotation_path):
    print()
    print(f"-------------------------- CONVERT {coco_split_name.upper()} --------------------------")

    with open(annotation_path, "r", encoding="utf-8") as f:
        coco = json.load(f)

    images = coco["images"]
    annotations = coco["annotations"]

    image_id_to_info = {
        int(image["id"]): image
        for image in images
    }

    anns_by_image_id = {}

    for ann in annotations:
        image_id = int(ann["image_id"])

        if image_id not in anns_by_image_id:
            anns_by_image_id[image_id] = []

        anns_by_image_id[image_id].append(ann)

    image_output_dir = YOLO_IMAGES_ROOT / yolo_split_name
    label_output_dir = YOLO_LABELS_ROOT / yolo_split_name

    converted_images = 0
    converted_annotations = 0
    skipped_annotations = 0

    for image_id, image_info in image_id_to_info.items():
        file_name = image_info["file_name"]
        width = int(image_info["width"])
        height = int(image_info["height"])

        src_image_path = image_dir / file_name

        if not src_image_path.exists():
            print("[WARNING] image not found:", src_image_path)
            continue

        dst_image_name = f"{image_id}_{Path(file_name).name}"
        dst_image_path = image_output_dir / dst_image_name

        shutil.copy2(src_image_path, dst_image_path)

        label_path = label_output_dir / f"{Path(dst_image_name).stem}.txt"

        label_lines = []

        for ann in anns_by_image_id.get(image_id, []):
            category_id = int(ann["category_id"])

            polygon = get_annotation_polygon(
                annotation=ann,
                image_width=width,
                image_height=height,
            )

            if polygon is None:
                skipped_annotations += 1
                continue

            normalized_polygon = normalize_polygon(
                polygon=polygon,
                width=width,
                height=height,
            )

            if normalized_polygon is None:
                skipped_annotations += 1
                continue

            values = [str(category_id)]
            values.extend([f"{v:.6f}" for v in normalized_polygon])

            label_lines.append(" ".join(values))
            converted_annotations += 1

        with open(label_path, "w", encoding="utf-8") as f:
            if len(label_lines) > 0:
                f.write("\n".join(label_lines))

        converted_images += 1

    print("converted_images:", converted_images)
    print("converted_annotations:", converted_annotations)
    print("skipped_annotations:", skipped_annotations)


def write_data_yaml():
    with open(TRAIN_ANN_PATH, "r", encoding="utf-8") as f:
        coco = json.load(f)

    categories = sorted(coco["categories"], key=lambda x: int(x["id"]))

    lines = []
    lines.append(f"path: {YOLO_DATASET_ROOT.as_posix()}")
    lines.append("train: images/train")
    lines.append("val: images/val")
    lines.append("test: images/test")
    lines.append("")
    lines.append("names:")

    for category in categories:
        category_id = int(category["id"])
        category_name = category["name"]
        lines.append(f"  {category_id}: {category_name}")

    with open(YOLO_DATA_YAML_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print()
    print("-------------------------- DATA YAML --------------------------")
    print("saved:", YOLO_DATA_YAML_PATH)
    print("\n".join(lines))


def convert_coco_to_yolo():
    make_dirs()

    convert_one_split(
        coco_split_name="train",
        yolo_split_name="train",
        image_dir=TRAIN_DIR,
        annotation_path=TRAIN_ANN_PATH,
    )

    convert_one_split(
        coco_split_name="valid",
        yolo_split_name="val",
        image_dir=VALID_DIR,
        annotation_path=VALID_ANN_PATH,
    )

    convert_one_split(
        coco_split_name="test",
        yolo_split_name="test",
        image_dir=TEST_DIR,
        annotation_path=TEST_ANN_PATH,
    )

    write_data_yaml()


# ============================================================
# 4. YOLO segmentation 학습
# ============================================================

def load_yolo_model():
    last_error = None

    for model_name in MODEL_CANDIDATES:
        try:
            print()
            print("-------------------------- MODEL INIT --------------------------")
            print("try model:", model_name)

            model = YOLO(model_name)

            print("loaded model:", model_name)
            return model

        except Exception as e:
            print("[WARNING] failed:", model_name)
            print(e)
            last_error = e

    raise RuntimeError(f"YOLO model load failed. last_error={last_error}")


def train_yolo_segmentation():
    print()
    print("-------------------------- TRAIN YOLO SEGMENTATION --------------------------")
    print("data:", YOLO_DATA_YAML_PATH)
    print("epochs:", EPOCHS)
    print("imgsz:", IMGSZ)
    print("batch:", BATCH)
    print("device:", DEVICE)

    model = load_yolo_model()

    model.train(
        data=str(YOLO_DATA_YAML_PATH),
        task="segment",
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        patience=PATIENCE,
        device=DEVICE,
        project=str(RUNS_ROOT),
        name=RUN_NAME,
        exist_ok=True,
    )

    best_model_path = RUNS_ROOT / RUN_NAME / "weights" / "best.pt"

    print()
    print("-------------------------- TRAIN DONE --------------------------")
    print("best model:", best_model_path)
    print("exists:", best_model_path.exists())


# ============================================================
# 5. main
# ============================================================

def main():
    check_paths()
    convert_coco_to_yolo()
    train_yolo_segmentation()


if __name__ == "__main__":
    main()