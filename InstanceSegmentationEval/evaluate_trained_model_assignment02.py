
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from pycocotools import mask as mask_utils
from ultralytics import YOLO

from pretrained_inference_eval_assignment01 import (
    PROJECT_ROOT,
    VALID_DIR,
    TEST_DIR,
    VALID_ANN_PATH,
    TEST_ANN_PATH,
    RESULT_DIR,
    evaluate_split,
)


# ============================================================
# 1. 설정
# ============================================================

RUNS_ROOT = PROJECT_ROOT / "runs_assignment02"
RUN_NAME = "yolo_seg_assignment02"

TRAINED_MODEL_PATH = RUNS_ROOT / RUN_NAME / "weights" / "best.pt"

VALID_YOLO_PRED_PATH = RESULT_DIR / "valid_yolo_assignment02_predictions.json"
TEST_YOLO_PRED_PATH = RESULT_DIR / "test_yolo_assignment02_predictions.json"

VALID_YOLO_CLASS_METRIC_PATH = RESULT_DIR / "valid_yolo_assignment02_class_metrics.csv"
TEST_YOLO_CLASS_METRIC_PATH = RESULT_DIR / "test_yolo_assignment02_class_metrics.csv"

SUMMARY_PATH = RESULT_DIR / "summary_assignment02_yolo.csv"

BASELINE_VALID_MAP50 = 0.532781
BASELINE_TEST_MAP50 = 0.567696

CONF_THRES = 0.001
IOU_THRES = 0.5
IMGSZ = 640

DEVICE = 0 if torch.cuda.is_available() else "cpu"


# ============================================================
# 2. mask encoding
# ============================================================

def encode_binary_mask(binary_mask):
    binary_mask = np.asfortranarray(binary_mask.astype(np.uint8))
    rle = mask_utils.encode(binary_mask)
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle


# ============================================================
# 3. YOLO prediction → COCO result JSON
# ============================================================

def run_yolo_inference_to_coco_json(model, image_dir, annotation_path, output_path, title):
    print()
    print(f"-------------------------- {title} YOLO INFERENCE --------------------------")

    with open(annotation_path, "r", encoding="utf-8") as f:
        coco = json.load(f)

    images = coco["images"]
    categories = coco["categories"]
    coco_category_ids = set(int(cat["id"]) for cat in categories)

    predictions = []

    for idx, image_info in enumerate(images):
        image_id = int(image_info["id"])
        file_name = image_info["file_name"]

        image_path = image_dir / file_name

        if not image_path.exists():
            print("[WARNING] image not found:", image_path)
            continue

        pil_image = Image.open(image_path).convert("RGB")
        original_width, original_height = pil_image.size

        results = model.predict(
            source=str(image_path),
            task="segment",
            conf=CONF_THRES,
            iou=IOU_THRES,
            imgsz=IMGSZ,
            device=DEVICE,
            retina_masks=True,
            verbose=False,
        )

        if len(results) == 0:
            continue

        result = results[0]

        if result.masks is None or result.boxes is None:
            continue

        masks = result.masks.data.cpu().numpy()
        boxes = result.boxes.xyxy.cpu().numpy()
        class_ids = result.boxes.cls.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()

        for mask, box, class_id, score in zip(masks, boxes, class_ids, confidences):
            category_id = int(class_id)

            if category_id not in coco_category_ids:
                continue

            binary_mask = mask.astype(np.uint8)

            if binary_mask.shape[0] != original_height or binary_mask.shape[1] != original_width:
                binary_mask = cv2.resize(
                    binary_mask,
                    (original_width, original_height),
                    interpolation=cv2.INTER_NEAREST,
                )

            if binary_mask.max() == 0:
                continue

            rle = encode_binary_mask(binary_mask)

            x, y, w, h = mask_utils.toBbox(rle).tolist()

            predictions.append({
                "image_id": image_id,
                "category_id": int(category_id),
                "bbox": [
                    float(x),
                    float(y),
                    float(w),
                    float(h),
                ],
                "score": float(score),
                "segmentation": rle,
            })

        if (idx + 1) % 20 == 0:
            print(f"{idx + 1}/{len(images)} images processed")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(predictions, f)

    print()
    print(f"{title} prediction saved:", output_path)
    print(f"{title} num predictions:", len(predictions))


# ============================================================
# 4. 평가
# ============================================================

def save_summary(valid_map50, test_map50):
    rows = [
        {
            "model": "baseline_IS_pretrained_bottom",
            "valid_map50": BASELINE_VALID_MAP50,
            "test_map50": BASELINE_TEST_MAP50,
        },
        {
            "model": "yolo_seg_assignment02",
            "valid_map50": valid_map50,
            "test_map50": test_map50,
        },
    ]

    with open(SUMMARY_PATH, "w", encoding="utf-8-sig") as f:
        f.write("model,valid_map50,test_map50\n")

        for row in rows:
            f.write(
                f'{row["model"]},'
                f'{row["valid_map50"]},'
                f'{row["test_map50"]}\n'
            )

    print()
    print("-------------------------- SUMMARY --------------------------")
    print("baseline valid mAP@50:", BASELINE_VALID_MAP50)
    print("baseline test  mAP@50:", BASELINE_TEST_MAP50)
    print("YOLO valid     mAP@50:", valid_map50)
    print("YOLO test      mAP@50:", test_map50)
    print("summary saved:", SUMMARY_PATH)

    print()
    if test_map50 > BASELINE_TEST_MAP50:
        print("RESULT: SUCCESS - YOLO model is better than baseline on test mAP@50.")
    else:
        print("RESULT: NOT YET - YOLO model did not exceed baseline test mAP@50.")


def main():
    print()
    print("-------------------------- EVALUATE TRAINED YOLO MODEL --------------------------")
    print("model:", TRAINED_MODEL_PATH)
    print("exists:", TRAINED_MODEL_PATH.exists())
    print("device:", DEVICE)

    if not TRAINED_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"trained model not found: {TRAINED_MODEL_PATH}\n"
            f"먼저 train_instance_segmentation_assignment02.py를 실행하세요."
        )

    model = YOLO(str(TRAINED_MODEL_PATH))

    run_yolo_inference_to_coco_json(
        model=model,
        image_dir=VALID_DIR,
        annotation_path=VALID_ANN_PATH,
        output_path=VALID_YOLO_PRED_PATH,
        title="VALID",
    )

    run_yolo_inference_to_coco_json(
        model=model,
        image_dir=TEST_DIR,
        annotation_path=TEST_ANN_PATH,
        output_path=TEST_YOLO_PRED_PATH,
        title="TEST",
    )

    valid_map50 = evaluate_split(
        annotation_path=VALID_ANN_PATH,
        prediction_path=VALID_YOLO_PRED_PATH,
        class_metric_path=VALID_YOLO_CLASS_METRIC_PATH,
        title="VALID YOLO ASSIGNMENT02",
    )

    test_map50 = evaluate_split(
        annotation_path=TEST_ANN_PATH,
        prediction_path=TEST_YOLO_PRED_PATH,
        class_metric_path=TEST_YOLO_CLASS_METRIC_PATH,
        title="TEST YOLO ASSIGNMENT02",
    )

    save_summary(
        valid_map50=valid_map50,
        test_map50=test_map50,
    )


if __name__ == "__main__":
    main()