
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from pycocotools import mask as mask_utils
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from config import IOU_THRES


# ============================================================
# 1. 공통 mask 처리 함수
# ============================================================

def _segmentation_to_rle(segmentation, height, width):
    """
    COCO annotation 또는 prediction의 segmentation을 RLE 형식으로 변환한다.
    polygon, uncompressed RLE, compressed RLE 모두 처리한다.
    """

    if isinstance(segmentation, list):
        rles = mask_utils.frPyObjects(segmentation, height, width)
        rle = mask_utils.merge(rles)
        return rle

    if isinstance(segmentation, dict):
        if isinstance(segmentation.get("counts"), list):
            rle = mask_utils.frPyObjects(segmentation, height, width)
            return rle

        rle = dict(segmentation)

        if isinstance(rle.get("counts"), str):
            rle["counts"] = rle["counts"].encode("utf-8")

        return rle

    return None


def _mask_iou(rle_a, rle_b):
    """
    두 RLE mask 사이의 IoU를 계산한다.
    """

    mask_a = mask_utils.decode(rle_a).astype(bool)
    mask_b = mask_utils.decode(rle_b).astype(bool)

    intersection = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()

    if union == 0:
        return 0.0

    return float(intersection / union)


# ============================================================
# 2. class별 AP@50 계산
# ============================================================

def evaluate_class_ap50(annotation_path, prediction_path):
    """
    COCOeval 결과에서 class별 AP@50을 계산한다.

    반환 항목:
    - category_id
    - class_name
    - gt_count
    - pred_count
    - ap50
    """

    coco_gt = COCO(str(annotation_path))

    with open(prediction_path, "r", encoding="utf-8") as f:
        predictions = json.load(f)

    category_ids = sorted(coco_gt.getCatIds())
    categories = coco_gt.loadCats(category_ids)
    category_id_to_name = {int(cat["id"]): cat["name"] for cat in categories}

    gt_count_by_category = defaultdict(int)
    pred_count_by_category = defaultdict(int)

    for ann in coco_gt.dataset.get("annotations", []):
        gt_count_by_category[int(ann["category_id"])] += 1

    for pred in predictions:
        pred_count_by_category[int(pred["category_id"])] += 1

    category_id_to_ap = {category_id: None for category_id in category_ids}

    if len(predictions) > 0:
        coco_dt = coco_gt.loadRes(predictions)

        coco_eval = COCOeval(coco_gt, coco_dt, "segm")
        coco_eval.params.iouThrs = np.array([IOU_THRES])
        coco_eval.params.catIds = category_ids
        coco_eval.evaluate()
        coco_eval.accumulate()

        # precision shape: [T, R, K, A, M]
        precision = coco_eval.eval["precision"]

        for category_index, category_id in enumerate(category_ids):
            # T=0: IoU 0.50, A=0: all area, M=-1: maxDets 100
            values = precision[0, :, category_index, 0, -1]
            values = values[values > -1]

            if len(values) == 0:
                category_id_to_ap[category_id] = None
            else:
                category_id_to_ap[category_id] = float(np.mean(values))

    results = []

    for category_id in category_ids:
        gt_count = int(gt_count_by_category[category_id])
        pred_count = int(pred_count_by_category[category_id])
        ap50 = category_id_to_ap[category_id]

        # GT가 없는 class는 AP를 N/A로 표시한다.
        if gt_count == 0:
            ap50 = None

        results.append({
            "category_id": int(category_id),
            "class_name": category_id_to_name[category_id],
            "gt_count": gt_count,
            "pred_count": pred_count,
            "ap50": ap50,
        })

    return results


# ============================================================
# 3. class별 Precision / Recall 계산
# ============================================================

def evaluate_class_precision_recall(annotation_path, prediction_path):
    """
    class별 TP, FP, FN, Precision, Recall을 계산한다.

    계산 방식:
    - image_id와 category_id가 같은 GT와 prediction끼리 비교한다.
    - prediction은 confidence score가 높은 순서대로 matching한다.
    - IoU가 0.5 이상이고 아직 matching되지 않은 GT가 있으면 TP로 본다.
    - matching 실패 prediction은 FP로 본다.
    - matching되지 않은 GT는 FN으로 본다.
    """

    coco_gt = COCO(str(annotation_path))

    with open(prediction_path, "r", encoding="utf-8") as f:
        predictions = json.load(f)

    category_ids = sorted(coco_gt.getCatIds())
    categories = coco_gt.loadCats(category_ids)
    category_id_to_name = {int(cat["id"]): cat["name"] for cat in categories}

    image_info_by_id = {
        int(image["id"]): image
        for image in coco_gt.dataset.get("images", [])
    }

    gt_by_image_category = defaultdict(list)
    pred_by_image_category = defaultdict(list)

    for ann in coco_gt.dataset.get("annotations", []):
        image_id = int(ann["image_id"])
        category_id = int(ann["category_id"])
        image_info = image_info_by_id[image_id]

        rle = _segmentation_to_rle(
            segmentation=ann["segmentation"],
            height=int(image_info["height"]),
            width=int(image_info["width"]),
        )

        if rle is None:
            continue

        gt_by_image_category[(image_id, category_id)].append({
            "rle": rle,
            "matched": False,
        })

    for pred in predictions:
        image_id = int(pred["image_id"])
        category_id = int(pred["category_id"])

        if image_id not in image_info_by_id:
            continue

        image_info = image_info_by_id[image_id]

        rle = _segmentation_to_rle(
            segmentation=pred["segmentation"],
            height=int(image_info["height"]),
            width=int(image_info["width"]),
        )

        if rle is None:
            continue

        pred_by_image_category[(image_id, category_id)].append({
            "rle": rle,
            "score": float(pred.get("score", 0.0)),
        })

    result_by_category = {}

    for category_id in category_ids:
        result_by_category[category_id] = {
            "category_id": int(category_id),
            "class_name": category_id_to_name[category_id],
            "tp": 0,
            "fp": 0,
            "fn": 0,
            "precision": None,
            "recall": None,
        }

    all_keys = set(gt_by_image_category.keys()) | set(pred_by_image_category.keys())

    for image_id, category_id in all_keys:
        gt_list = gt_by_image_category.get((image_id, category_id), [])
        pred_list = pred_by_image_category.get((image_id, category_id), [])

        pred_list = sorted(pred_list, key=lambda x: x["score"], reverse=True)

        for pred in pred_list:
            best_iou = 0.0
            best_gt_index = -1

            for gt_index, gt in enumerate(gt_list):
                if gt["matched"]:
                    continue

                iou = _mask_iou(pred["rle"], gt["rle"])

                if iou > best_iou:
                    best_iou = iou
                    best_gt_index = gt_index

            if best_iou >= IOU_THRES and best_gt_index >= 0:
                gt_list[best_gt_index]["matched"] = True
                result_by_category[category_id]["tp"] += 1
            else:
                result_by_category[category_id]["fp"] += 1

        for gt in gt_list:
            if not gt["matched"]:
                result_by_category[category_id]["fn"] += 1

    for category_id in category_ids:
        row = result_by_category[category_id]
        tp = row["tp"]
        fp = row["fp"]
        fn = row["fn"]

        if tp + fp > 0:
            row["precision"] = tp / (tp + fp)

        if tp + fn > 0:
            row["recall"] = tp / (tp + fn)

    return [result_by_category[category_id] for category_id in category_ids]


# ============================================================
# 4. class별 metric CSV 저장
# ============================================================

def save_class_metrics_csv(class_ap_results, pr_results, output_csv_path, title):
    """
    class별 AP@50, TP, FP, FN, Precision, Recall을 CSV로 저장하고 콘솔에 출력한다.
    """

    pr_by_category_id = {
        int(row["category_id"]): row
        for row in pr_results
    }

    output_csv_path = Path(output_csv_path)
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow([
            "category_id",
            "class_name",
            "gt_count",
            "pred_count",
            "ap50",
            "tp",
            "fp",
            "fn",
            "precision",
            "recall",
        ])

        for ap_row in class_ap_results:
            category_id = int(ap_row["category_id"])
            pr_row = pr_by_category_id[category_id]

            writer.writerow([
                category_id,
                ap_row["class_name"],
                ap_row["gt_count"],
                ap_row["pred_count"],
                "N/A" if ap_row["ap50"] is None else f'{ap_row["ap50"]:.6f}',
                pr_row["tp"],
                pr_row["fp"],
                pr_row["fn"],
                "N/A" if pr_row["precision"] is None else f'{pr_row["precision"]:.6f}',
                "N/A" if pr_row["recall"] is None else f'{pr_row["recall"]:.6f}',
            ])

    print()
    print(f"-------------------------- {title} CLASS METRICS --------------------------")
    print("saved:", output_csv_path)
    print()
    print("category_id | class_name        | GT | PRED | AP@50    | TP | FP | FN | Precision | Recall")
    print("-" * 100)

    for ap_row in class_ap_results:
        category_id = int(ap_row["category_id"])
        pr_row = pr_by_category_id[category_id]

        ap_text = "N/A" if ap_row["ap50"] is None else f'{ap_row["ap50"]:.6f}'
        precision_text = "N/A" if pr_row["precision"] is None else f'{pr_row["precision"]:.6f}'
        recall_text = "N/A" if pr_row["recall"] is None else f'{pr_row["recall"]:.6f}'

        print(
            f'{category_id:11d} | '
            f'{ap_row["class_name"]:<17s} | '
            f'{ap_row["gt_count"]:2d} | '
            f'{ap_row["pred_count"]:4d} | '
            f'{ap_text:<8s} | '
            f'{pr_row["tp"]:2d} | '
            f'{pr_row["fp"]:2d} | '
            f'{pr_row["fn"]:2d} | '
            f'{precision_text:<9s} | '
            f'{recall_text}'
        )
