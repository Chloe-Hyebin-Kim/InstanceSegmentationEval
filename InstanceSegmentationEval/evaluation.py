
import json

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from config import IOU_THRES
from metrics import (
    evaluate_class_ap50,
    evaluate_class_precision_recall,
    save_class_metrics_csv,
)


# ============================================================
# 1. COCOeval mAP@50 계산
# ============================================================

def evaluate_map50(annotation_path, prediction_path, title):
    """
    COCOeval을 사용하여 segmentation mAP@50을 계산한다.

    평가 조건:
    - annotation type: segm
    - IoU threshold: 0.50
    - area: all
    - maxDets: 100
    """

    print()
    print(f"-------------------------- {title} mAP@50 --------------------------")

    with open(prediction_path, "r", encoding="utf-8") as f:
        predictions = json.load(f)

    if len(predictions) == 0:
        print("prediction result is empty. mAP@50 = 0.0")
        return 0.0

    coco_gt = COCO(str(annotation_path))
    coco_dt = coco_gt.loadRes(predictions)

    coco_eval = COCOeval(coco_gt, coco_dt, "segm")
    coco_eval.params.iouThrs = np.array([IOU_THRES])

    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    map50 = float(coco_eval.stats[0])
    print(f"{title} segm mAP@50 = {map50:.6f}")

    return map50


# ============================================================
# 2. 전체 평가 흐름
# ============================================================

def evaluate_split(annotation_path, prediction_path, class_metric_path, title):
    """
    valid 또는 test split 하나에 대해 전체 평가를 수행한다.

    수행 내용:
    1) COCOeval segmentation mAP@50 계산
    2) class별 AP@50 계산
    3) class별 TP/FP/FN/Precision/Recall 계산
    4) class별 metric을 CSV로 저장
    """

    map50 = evaluate_map50(
        annotation_path=annotation_path,
        prediction_path=prediction_path,
        title=title,
    )

    class_ap_results = evaluate_class_ap50(
        annotation_path=annotation_path,
        prediction_path=prediction_path,
    )

    pr_results = evaluate_class_precision_recall(
        annotation_path=annotation_path,
        prediction_path=prediction_path,
    )

    save_class_metrics_csv(
        class_ap_results=class_ap_results,
        pr_results=pr_results,
        output_csv_path=class_metric_path,
        title=title,
    )

    return map50
