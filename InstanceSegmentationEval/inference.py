
import json

import cv2
import numpy as np
from PIL import Image
from pycocotools import mask as mask_utils

from config import (
    RFDETR_CONF_THRES,
    YOLO_CONF_THRES,
    IOU_THRES,
    IMGSZ,
    DEVICE,
)
from utils import encode_binary_mask


# ============================================================
# 1. RF-DETR prediction → COCO result JSON
# ============================================================

def run_rfdetr_inference_to_coco_json(model, image_dir, annotation_path, output_path, title):
    """
    제공된 pretrained RF-DETR segmentation 모델로 valid/test 이미지를 추론하고,
    COCOeval에서 사용할 수 있는 prediction JSON을 저장한다.

    중요한 점:
    - RF-DETR의 raw class_id를 COCO category_id로 그대로 사용한다.
    - 이전 실험에서 class name 기반으로 다시 매핑하면 category가 밀려 mAP가 0이 될 수 있었다.
    - 따라서 class_id → category_id는 int(class_id) 그대로 사용한다.
    """

    print()
    print(f"-------------------------- {title} INFERENCE --------------------------")

    with open(annotation_path, "r", encoding="utf-8") as f:
        coco = json.load(f)

    images = coco["images"]
    categories = coco["categories"]
    coco_category_ids = set(int(cat["id"]) for cat in categories)

    predictions = []
    skipped_no_mask = 0
    skipped_invalid_category = 0

    for idx, image_info in enumerate(images):
        image_id = int(image_info["id"])
        file_name = image_info["file_name"]

        image_path = image_dir / file_name

        if not image_path.exists():
            print("[WARNING] image not found:", image_path)
            continue

        image = Image.open(image_path).convert("RGB")
        original_width, original_height = image.size

        detections = model.predict(image, threshold=RFDETR_CONF_THRES)

        masks = getattr(detections, "mask", None)
        class_ids = getattr(detections, "class_id", None)
        confidences = getattr(detections, "confidence", None)

        if masks is None or class_ids is None or confidences is None:
            skipped_no_mask += 1
            continue

        masks = np.asarray(masks)
        class_ids = np.asarray(class_ids)
        confidences = np.asarray(confidences)

        for mask, class_id, score in zip(masks, class_ids, confidences):
            category_id = int(class_id)

            if category_id not in coco_category_ids:
                skipped_invalid_category += 1
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
                "category_id": category_id,
                "bbox": [float(x), float(y), float(w), float(h)],
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
    print(f"{title} skipped_no_mask images:", skipped_no_mask)
    print(f"{title} skipped_invalid_category predictions:", skipped_invalid_category)


# ============================================================
# 2. YOLO prediction → COCO result JSON
# ============================================================

def run_yolo_inference_to_coco_json(model, image_dir, annotation_path, output_path, title):
    """
    fine-tuning된 YOLO segmentation 모델로 valid/test 이미지를 추론하고,
    결과를 COCOeval용 prediction JSON으로 저장한다.

    YOLO 내부 prediction 형식:
    - result.masks.data
    - result.boxes.cls
    - result.boxes.conf

    COCOeval prediction 형식:
    - image_id
    - category_id
    - bbox: [x, y, width, height]
    - score
    - segmentation: RLE
    """

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
            conf=YOLO_CONF_THRES,
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
        class_ids = result.boxes.cls.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()

        for mask, class_id, score in zip(masks, class_ids, confidences):
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
                "bbox": [float(x), float(y), float(w), float(h)],
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
