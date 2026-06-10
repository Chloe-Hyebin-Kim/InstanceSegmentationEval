
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


PROJECT_ROOT = Path.cwd()

VALID_ANN_PATH = PROJECT_ROOT / "dataset-coco-seg" / "dataset-coco-seg" / "valid" / "_annotations.coco.json"
TEST_ANN_PATH = PROJECT_ROOT / "dataset-coco-seg" / "dataset-coco-seg" / "test" / "_annotations.coco.json"

VALID_PRED_PATH = PROJECT_ROOT / "valid_predictions.json"
TEST_PRED_PATH = PROJECT_ROOT / "test_predictions.json"

OUTPUT_CSV_PATH = PROJECT_ROOT / "class_ap50_results.csv"


def evaluate_class_ap50(split_name, ann_path, pred_path):
    print()
    print(f"-------------------------- {split_name.upper()} CLASS AP@50 --------------------------")
    print("annotation exists:", ann_path.exists())
    print("prediction exists:", pred_path.exists())

    with open(ann_path, "r", encoding="utf-8") as f:
        ann_data = json.load(f)

    with open(pred_path, "r", encoding="utf-8") as f:
        pred_data = json.load(f)

    categories = ann_data["categories"]

    category_names = {
        c["id"]: c["name"]
        for c in categories
    }

    gt_count = Counter(
        ann["category_id"]
        for ann in ann_data["annotations"]
    )

    pred_count = Counter(
        pred["category_id"]
        for pred in pred_data
    )

    coco_gt = COCO(str(ann_path))
    coco_pred = coco_gt.loadRes(str(pred_path))

    evaluator = COCOeval(coco_gt, coco_pred, iouType="segm")

    evaluator.params.iouThrs = np.array([0.5])
    evaluator.params.maxDets = [1, 10, 500]

    evaluator.evaluate()
    evaluator.accumulate()

    precision = evaluator.eval["precision"]

    # [IoU threshold, recall threshold, category, area range, max dets]
    iou_index = 0
    area_index = 0
    max_det_index = 2

    results = []

    print()
    print(f"{'category_id':>11} | {'class_name':<15} | {'GT':>4} | {'PRED':>5} | {'AP@50':>7}")
    print("-" * 58)

    for category_index, category in enumerate(categories):
        category_id = category["id"]
        class_name = category["name"]

        precision_values = precision[
            iou_index,
            :,
            category_index,
            area_index,
            max_det_index
        ]

        valid_precision_values = precision_values[precision_values > -1]

        if len(valid_precision_values) == 0:
            ap50 = None
            ap50_text = "N/A"
        else:
            ap50 = float(np.mean(valid_precision_values))
            ap50_text = f"{ap50:.3f}"

        gt_num = gt_count.get(category_id, 0)
        pred_num = pred_count.get(category_id, 0)

        print(f"{category_id:>11} | {class_name:<15} | {gt_num:>4} | {pred_num:>5} | {ap50_text:>7}")

        results.append({
            "split": split_name,
            "category_id": category_id,
            "class_name": class_name,
            "gt_count": gt_num,
            "pred_count": pred_num,
            "ap50": "" if ap50 is None else ap50,
        })

    return results


all_results = []

all_results.extend(
    evaluate_class_ap50(
        split_name="valid",
        ann_path=VALID_ANN_PATH,
        pred_path=VALID_PRED_PATH
    )
)

all_results.extend(
    evaluate_class_ap50(
        split_name="test",
        ann_path=TEST_ANN_PATH,
        pred_path=TEST_PRED_PATH
    )
)


with open(OUTPUT_CSV_PATH, "w", newline="", encoding="utf-8-sig") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "split",
            "category_id",
            "class_name",
            "gt_count",
            "pred_count",
            "ap50",
        ]
    )

    writer.writeheader()
    writer.writerows(all_results)

print()
print("-------------------------- SAVE RESULT --------------------------")
print("saved:", OUTPUT_CSV_PATH)


#Validation mAP@50 = 0.616
#Test mAP@50       = 0.592


#성능 좋은 클래스
#test quarter AP@50  = 1.000
#valid hem AP@50     = 0.906
#valid logo AP@50    = 0.904
#valid quarter AP@50 = 0.862
#test logo AP@50     = 0.834


#성능 나쁜 클래스
#valid decoration AP@50 = 0.000
#valid lace AP@50       = 0.000
#valid zipper AP@50     = 0.000
#test zipper AP@50      = 0.000
#test pocket AP@50      = 0.274
#test decoration AP@50  = 0.484
#valid pocket AP@50     = 0.476


#예측이 너무 많음 
#면적과 성능 비례?
#클래스 불균형
