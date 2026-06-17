

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
from pycocotools import mask as mask_utils
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from rfdetr import RFDETRSegMedium


# ============================================================
# 1. PATH 설정
# ============================================================

PROJECT_ROOT = Path(r"D:\git\InstanceSegmentationEval\InstanceSegmentationEval")

MODEL_PATH = PROJECT_ROOT / "IS_pretrained_bottom.pt"

DATASET_ROOT = PROJECT_ROOT / "dataset-coco-seg"

VALID_DIR = DATASET_ROOT / "valid"
TEST_DIR = DATASET_ROOT / "test"

VALID_ANN_PATH = VALID_DIR / "_annotations.coco.json"
TEST_ANN_PATH = TEST_DIR / "_annotations.coco.json"

RESULT_DIR = PROJECT_ROOT / "results"
RESULT_DIR.mkdir(exist_ok=True)

VALID_PRED_PATH = RESULT_DIR / "valid_predictions.json"
TEST_PRED_PATH = RESULT_DIR / "test_predictions.json"

VALID_CLASS_METRIC_PATH = RESULT_DIR / "valid_class_metrics.csv"
TEST_CLASS_METRIC_PATH = RESULT_DIR / "test_class_metrics.csv"
SUMMARY_PATH = RESULT_DIR / "summary_map50.csv"


# ============================================================
# 2. 평가 설정
# ============================================================

CONF_THRES = 0.1
IOU_THRES = 0.5

# 모델 class_id와 COCO category_id가 다를 경우 여기에서 직접 수정
# 예:
# CATEGORY_ID_MAPPING = {
#     0: 1,
#     1: 2,
#     2: 3,
# }
CATEGORY_ID_MAPPING = {}


# ============================================================
# 3. 경로 확인
# ============================================================

def check_paths():
    print()
    print("-------------------------- PATH CHECK --------------------------")
    print("PROJECT_ROOT:", PROJECT_ROOT)
    print("MODEL_PATH:", MODEL_PATH)
    print("DATASET_ROOT:", DATASET_ROOT)
    print()

    path_items = {
        "MODEL exists": MODEL_PATH,
        "VALID_DIR exists": VALID_DIR,
        "TEST_DIR exists": TEST_DIR,
        "VALID_ANN exists": VALID_ANN_PATH,
        "TEST_ANN exists": TEST_ANN_PATH,
    }

    has_error = False

    for name, path in path_items.items():
        exists = path.exists()
        print(f"{name}: {exists} -> {path}")
        if not exists:
            has_error = True

    if has_error:
        raise FileNotFoundError("필수 파일 또는 폴더 경로가 잘못되었습니다.")

    print("PATH CHECK OK")


# ============================================================
# 4. mask / class_id 처리
# ============================================================

def encode_binary_mask(binary_mask):
    binary_mask = np.asfortranarray(binary_mask.astype(np.uint8))
    rle = mask_utils.encode(binary_mask)
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle


def get_model_class_name(model, class_id):
    class_names = getattr(model, "class_names", None)

    if class_names is None:
        return None

    if isinstance(class_names, dict):
        if class_id in class_names:
            return class_names[class_id]
        if str(class_id) in class_names:
            return class_names[str(class_id)]
        return None

    if isinstance(class_names, list):
        if 0 <= class_id < len(class_names):
            return class_names[class_id]
        return None

    return None

def map_category_id(raw_class_id):
    return int(raw_class_id)

    if raw_class_id in CATEGORY_ID_MAPPING:
        return int(CATEGORY_ID_MAPPING[raw_class_id])

    model_class_name = get_model_class_name(model, raw_class_id)

    if model_class_name is not None and model_class_name in coco_name_to_id:
        return int(coco_name_to_id[model_class_name])

    return raw_class_id


def print_dataset_categories(annotation_path, title):
    with open(annotation_path, "r", encoding="utf-8") as f:
        coco_json = json.load(f)

    print()
    print(f"-------------------------- {title} CATEGORIES --------------------------")
    for cat in coco_json["categories"]:
        print(f'id={cat["id"]}, name={cat["name"]}')


# ============================================================
# 5. inference
# ============================================================

def run_inference(model, image_dir, annotation_path, output_path, title):
    print()
    print(f"-------------------------- {title} INFERENCE --------------------------")

    with open(annotation_path, "r", encoding="utf-8") as f:
        coco_json = json.load(f)

    images = coco_json["images"]
    categories = coco_json["categories"]
    coco_category_ids = set(int(cat["id"]) for cat in categories)

    predictions = []
    skipped_invalid_category = 0
    skipped_no_mask = 0
    
    for idx, image_info in enumerate(images):
        image_id = int(image_info["id"])
        file_name = image_info["file_name"]
        image_path = image_dir / file_name
    
        if not image_path.exists():
            print(f"[WARNING] image not found: {image_path}")
            continue
    
        image = Image.open(image_path).convert("RGB")
    
        detections = model.predict(image, threshold=CONF_THRES)
    
        xyxy = getattr(detections, "xyxy", None)
        class_ids = getattr(detections, "class_id", None)
        confidences = getattr(detections, "confidence", None)
        masks = getattr(detections, "mask", None)
    
        if xyxy is None or class_ids is None or confidences is None:
            continue
    
        if masks is None:
            skipped_no_mask += 1
            continue
    
        xyxy = np.asarray(xyxy)
        class_ids = np.asarray(class_ids)
        confidences = np.asarray(confidences)
        masks = np.asarray(masks)
    
        for box, raw_class_id, score, mask in zip(xyxy, class_ids, confidences, masks):
            category_id = int(raw_class_id)
    
            if category_id not in coco_category_ids:
                skipped_invalid_category += 1
                continue
    
            binary_mask = mask.astype(np.uint8)
    
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
    print(f"{title} prediction saved: {output_path}")
    print(f"{title} num predictions: {len(predictions)}")
    print(f"{title} skipped_no_mask images: {skipped_no_mask}")
    print(f"{title} skipped_invalid_category predictions: {skipped_invalid_category}")


# ============================================================
# 6. 전체 mAP@50 평가
# ============================================================

def evaluate_map50(annotation_path, prediction_path, title):
    print()
    print(f"-------------------------- {title} mAP@50 --------------------------")

    with open(prediction_path, "r", encoding="utf-8") as f:
        predictions = json.load(f)

    if len(predictions) == 0:
        print(f"{title} prediction이 0개입니다. mAP@50 = 0.0")
        return 0.0

    coco_gt = COCO(str(annotation_path))
    coco_dt = coco_gt.loadRes(str(prediction_path))

    coco_eval = COCOeval(coco_gt, coco_dt, "segm")
    coco_eval.params.iouThrs = np.array([IOU_THRES])
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    map50 = float(coco_eval.stats[0])
    print(f"{title} segm mAP@50 = {map50:.6f}")

    return map50


# ============================================================
# 7. 클래스별 AP@50
# ============================================================

def evaluate_class_ap50(annotation_path, prediction_path):
    coco_gt = COCO(str(annotation_path))

    with open(prediction_path, "r", encoding="utf-8") as f:
        predictions = json.load(f)

    cat_ids = coco_gt.getCatIds()
    cats = coco_gt.loadCats(cat_ids)

    if len(predictions) == 0:
        results = {}

        for cat in cats:
            cat_id = int(cat["id"])
            gt_count = len(coco_gt.getAnnIds(catIds=[cat_id]))
            results[cat_id] = {
                "category_id": cat_id,
                "class_name": cat["name"],
                "gt_count": gt_count,
                "pred_count": 0,
                "ap50": None if gt_count == 0 else 0.0,
            }

        return results

    coco_dt = coco_gt.loadRes(str(prediction_path))

    coco_eval = COCOeval(coco_gt, coco_dt, "segm")
    coco_eval.params.iouThrs = np.array([IOU_THRES])
    coco_eval.evaluate()
    coco_eval.accumulate()

    precision = coco_eval.eval["precision"]

    # precision shape:
    # [T, R, K, A, M]
    # T: IoU threshold
    # R: recall threshold
    # K: category
    # A: area range
    # M: max detections
    results = {}

    pred_count_by_cat = defaultdict(int)
    for pred in predictions:
        pred_count_by_cat[int(pred["category_id"])] += 1

    for cat_idx, cat in enumerate(cats):
        cat_id = int(cat["id"])
        class_name = cat["name"]

        gt_count = len(coco_gt.getAnnIds(catIds=[cat_id]))
        pred_count = pred_count_by_cat[cat_id]

        if gt_count == 0:
            ap50 = None
        else:
            p = precision[0, :, cat_idx, 0, -1]
            valid_p = p[p > -1]

            if len(valid_p) == 0:
                ap50 = 0.0
            else:
                ap50 = float(np.mean(valid_p))

        results[cat_id] = {
            "category_id": cat_id,
            "class_name": class_name,
            "gt_count": gt_count,
            "pred_count": pred_count,
            "ap50": ap50,
        }

    return results


# ============================================================
# 8. 클래스별 TP / FP / FN / Precision / Recall
# ============================================================

def make_pred_rle_for_iou(pred_segmentation):
    rle = {
        "size": pred_segmentation["size"],
        "counts": pred_segmentation["counts"],
    }

    if isinstance(rle["counts"], str):
        rle["counts"] = rle["counts"].encode("utf-8")

    return rle


def evaluate_class_precision_recall(annotation_path, prediction_path):
    coco_gt = COCO(str(annotation_path))

    with open(prediction_path, "r", encoding="utf-8") as f:
        predictions = json.load(f)

    cat_ids = coco_gt.getCatIds()
    cats = coco_gt.loadCats(cat_ids)

    metrics = {}

    for cat in cats:
        cat_id = int(cat["id"])
        metrics[cat_id] = {
            "category_id": cat_id,
            "class_name": cat["name"],
            "tp": 0,
            "fp": 0,
            "fn": 0,
            "precision": None,
            "recall": None,
        }

    gt_by_image_cat = defaultdict(list)
    pred_by_image_cat = defaultdict(list)

    ann_ids = coco_gt.getAnnIds()
    anns = coco_gt.loadAnns(ann_ids)

    for ann in anns:
        image_id = int(ann["image_id"])
        category_id = int(ann["category_id"])

        rle = coco_gt.annToRLE(ann)

        gt_by_image_cat[(image_id, category_id)].append({
            "ann_id": int(ann["id"]),
            "rle": rle,
            "matched": False,
        })

    for pred in predictions:
        image_id = int(pred["image_id"])
        category_id = int(pred["category_id"])

        pred_by_image_cat[(image_id, category_id)].append({
            "score": float(pred["score"]),
            "rle": make_pred_rle_for_iou(pred["segmentation"]),
        })

    all_keys = set(gt_by_image_cat.keys()) | set(pred_by_image_cat.keys())

    for key in all_keys:
        image_id, category_id = key

        gt_items = gt_by_image_cat.get(key, [])
        pred_items = pred_by_image_cat.get(key, [])

        pred_items = sorted(pred_items, key=lambda x: x["score"], reverse=True)

        if category_id not in metrics:
            continue

        for pred_item in pred_items:
            if len(gt_items) == 0:
                metrics[category_id]["fp"] += 1
                continue

            gt_rles = [gt_item["rle"] for gt_item in gt_items]
            iscrowd = [0 for _ in gt_items]

            ious = mask_utils.iou([pred_item["rle"]], gt_rles, iscrowd)[0]

            best_iou = -1.0
            best_gt_idx = -1

            for gt_idx, iou in enumerate(ious):
                if gt_items[gt_idx]["matched"]:
                    continue

                if iou > best_iou:
                    best_iou = float(iou)
                    best_gt_idx = gt_idx

            if best_iou >= IOU_THRES and best_gt_idx >= 0:
                gt_items[best_gt_idx]["matched"] = True
                metrics[category_id]["tp"] += 1
            else:
                metrics[category_id]["fp"] += 1

        for gt_item in gt_items:
            if not gt_item["matched"]:
                metrics[category_id]["fn"] += 1

    for cat_id, item in metrics.items():
        tp = item["tp"]
        fp = item["fp"]
        fn = item["fn"]

        if tp + fp > 0:
            item["precision"] = tp / (tp + fp)
        else:
            item["precision"] = None

        if tp + fn > 0:
            item["recall"] = tp / (tp + fn)
        else:
            item["recall"] = None

    return metrics


# ============================================================
# 9. 클래스별 결과 CSV 저장
# ============================================================

def save_class_metrics_csv(class_ap_results, pr_results, output_csv_path, title):
    rows = []

    for cat_id in sorted(class_ap_results.keys()):
        ap_item = class_ap_results[cat_id]
        pr_item = pr_results[cat_id]

        row = {
            "category_id": cat_id,
            "class_name": ap_item["class_name"],
            "gt_count": ap_item["gt_count"],
            "pred_count": ap_item["pred_count"],
            "ap50": ap_item["ap50"],
            "tp": pr_item["tp"],
            "fp": pr_item["fp"],
            "fn": pr_item["fn"],
            "precision": pr_item["precision"],
            "recall": pr_item["recall"],
        }

        rows.append(row)

    fieldnames = [
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
    ]

    with open(output_csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print()
    print(f"-------------------------- {title} CLASS METRICS --------------------------")
    print(f"saved: {output_csv_path}")
    print()
    print("category_id | class_name        | GT | PRED | AP@50    | TP | FP | FN | Precision | Recall")
    print("-" * 100)

    for row in rows:
        ap_text = "N/A" if row["ap50"] is None else f'{row["ap50"]:.6f}'
        precision_text = "N/A" if row["precision"] is None else f'{row["precision"]:.6f}'
        recall_text = "N/A" if row["recall"] is None else f'{row["recall"]:.6f}'

        print(
            f'{row["category_id"]:11d} | '
            f'{row["class_name"]:<17s} | '
            f'{row["gt_count"]:2d} | '
            f'{row["pred_count"]:4d} | '
            f'{ap_text:8s} | '
            f'{row["tp"]:2d} | '
            f'{row["fp"]:2d} | '
            f'{row["fn"]:2d} | '
            f'{precision_text:9s} | '
            f'{recall_text:6s}'
        )


# ============================================================
# 10. split별 평가
# ============================================================

def evaluate_split(annotation_path, prediction_path, class_metric_path, title):
    map50 = evaluate_map50(annotation_path, prediction_path, title)

    class_ap_results = evaluate_class_ap50(annotation_path, prediction_path)
    pr_results = evaluate_class_precision_recall(annotation_path, prediction_path)

    save_class_metrics_csv(
        class_ap_results=class_ap_results,
        pr_results=pr_results,
        output_csv_path=class_metric_path,
        title=title,
    )

    return map50


## ============================================================
## 11. main
## ============================================================

#def main():
#    check_paths()

#    print_dataset_categories(VALID_ANN_PATH, "VALID")
#    print_dataset_categories(TEST_ANN_PATH, "TEST")

#    print()
#    print("-------------------------- MODEL LOAD --------------------------")
#    model = RFDETRSegMedium.from_checkpoint(str(MODEL_PATH))
#    print("model loaded:", MODEL_PATH)

#    model_class_names = getattr(model, "class_names", None)
#    print("model.class_names:")
#    print(model_class_names)

#    run_inference(
#        model=model,
#        image_dir=VALID_DIR,
#        annotation_path=VALID_ANN_PATH,
#        output_path=VALID_PRED_PATH,
#        title="VALID",
#    )

#    run_inference(
#        model=model,
#        image_dir=TEST_DIR,
#        annotation_path=TEST_ANN_PATH,
#        output_path=TEST_PRED_PATH,
#        title="TEST",
#    )

#    valid_map50 = evaluate_split(
#        annotation_path=VALID_ANN_PATH,
#        prediction_path=VALID_PRED_PATH,
#        class_metric_path=VALID_CLASS_METRIC_PATH,
#        title="VALID",
#    )

#    test_map50 = evaluate_split(
#        annotation_path=TEST_ANN_PATH,
#        prediction_path=TEST_PRED_PATH,
#        class_metric_path=TEST_CLASS_METRIC_PATH,
#        title="TEST",
#    )

#    with open(SUMMARY_PATH, "w", newline="", encoding="utf-8-sig") as f:
#        writer = csv.DictWriter(f, fieldnames=["split", "map50", "conf_threshold", "iou_threshold"])
#        writer.writeheader()
#        writer.writerow({
#            "split": "valid",
#            "map50": valid_map50,
#            "conf_threshold": CONF_THRES,
#            "iou_threshold": IOU_THRES,
#        })
#        writer.writerow({
#            "split": "test",
#            "map50": test_map50,
#            "conf_threshold": CONF_THRES,
#            "iou_threshold": IOU_THRES,
#        })

#    print()
#    print("-------------------------- FINAL SUMMARY --------------------------")
#    print(f"VALID mAP@50: {valid_map50:.6f}")
#    print(f"TEST  mAP@50: {test_map50:.6f}")
#    print(f"summary saved: {SUMMARY_PATH}")


#if __name__ == "__main__":
#    main()