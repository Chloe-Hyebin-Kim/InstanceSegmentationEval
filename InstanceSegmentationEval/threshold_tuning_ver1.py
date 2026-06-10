
#confidence threshold
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image
from pycocotools import mask as mask_utils
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from rfdetr import RFDETRSegMedium


PROJECT_ROOT = Path.cwd()
MODEL_PATH = PROJECT_ROOT / "IS_pretrained_bottom.pt"
DATASET_ROOT = PROJECT_ROOT / "dataset-coco-seg" / "dataset-coco-seg"
VALID_DIR = DATASET_ROOT / "valid"
VALID_ANN_PATH = VALID_DIR / "_annotations.coco.json"

RESULT_CSV_PATH = PROJECT_ROOT / "threshold_tuning_ver1_results.csv"

THRESHOLDS = [0.05, 0.10, 0.20, 0.30, 0.50]

print("-------------------------- PATH CHECK --------------------------")
print("MODEL exists:", MODEL_PATH.exists())
print("VALID_ANN exists:", VALID_ANN_PATH.exists())

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"model file not found: {MODEL_PATH}")

if not VALID_ANN_PATH.exists():
    raise FileNotFoundError(f"valid annotation not found: {VALID_ANN_PATH}")


print("\n")
print("-------------------------- DATASET --------------------------")
with open(VALID_ANN_PATH, "r", encoding="utf-8") as f:
    valid_coco_data = json.load(f)

valid_images = valid_coco_data["images"]

print("valid image count:", len(valid_images))


print("\n")
print("-------------------------- LOAD MODEL --------------------------")
model = RFDETRSegMedium.from_checkpoint(str(MODEL_PATH))

print("model load success")



print("\n")
print(f"-------------------------- INFERENCE threshold={threshold:.2f} --------------------------")
def run_valid_inference(threshold):
    output_json_path = PROJECT_ROOT / f"valid_predictions_th{int(threshold * 100):03d}.json"

    predictions = []


    for idx, image_info in enumerate(valid_images, start=1):
        image_id = image_info["id"]
        file_name = image_info["file_name"]
        image_path = VALID_DIR / file_name

        print(f"[{idx}/{len(valid_images)}] inference: {file_name}")

        image = Image.open(image_path).convert("RGB")
        detections = model.predict(image, threshold=threshold)

        if detections.mask is None:
            continue

        for mask, class_id, score in zip(
            detections.mask,
            detections.class_id,
            detections.confidence
        ):
            mask = np.asfortranarray(mask.astype(np.uint8))

            rle = mask_utils.encode(mask)
            rle["counts"] = rle["counts"].decode("utf-8")

            predictions.append({
                "image_id": int(image_id),
                "category_id": int(class_id),  # +1 하지 않음
                "segmentation": rle,
                "score": float(score),
            })

    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(predictions, f)

    print("saved:", output_json_path)
    print("prediction count:", len(predictions))

    return output_json_path, len(predictions)

def evaluate_map50(pred_json_path):
    if not pred_json_path.exists():
        raise FileNotFoundError(f"prediction json not found: {pred_json_path}")

    with open(pred_json_path, "r", encoding="utf-8") as f:
        predictions = json.load(f)

    if len(predictions) == 0:
        return 0.0

    gt = COCO(str(VALID_ANN_PATH))
    pred = gt.loadRes(str(pred_json_path))

    evaluator = COCOeval(gt, pred, iouType="segm")
    evaluator.params.iouThrs = np.array([0.5])
    evaluator.params.maxDets = [1, 10, 500]

    evaluator.evaluate()
    evaluator.accumulate()

    precision = evaluator.eval["precision"]

    # [iou, recall, category, area, maxDets]
    precision_values = precision[0, :, :, 0, 2]
    valid_precision_values = precision_values[precision_values > -1]

    if len(valid_precision_values) == 0:
        return 0.0

    return float(np.mean(valid_precision_values))

results = []

for threshold in THRESHOLDS:
    pred_json_path, pred_count = run_valid_inference(threshold)
    map50 = evaluate_map50(pred_json_path)

    results.append({
        "threshold": threshold,
        "prediction_count": pred_count,
        "map50": map50,
        "prediction_json": pred_json_path.name,
    })

    print()
    print(f"threshold={threshold:.2f}, prediction_count={pred_count}, mAP@50={map50:.3f}")

with open(RESULT_CSV_PATH, "w", newline="", encoding="utf-8-sig") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "threshold",
            "prediction_count",
            "map50",
            "prediction_json",
        ]
    )

    writer.writeheader()
    writer.writerows(results)

best_result = max(results, key=lambda x: x["map50"])

print()
print("-------------------------- THRESHOLD TUNING RESULT --------------------------")

for result in results:
    print(
        f"threshold={result['threshold']:.2f} | "
        f"pred={result['prediction_count']:>4} | "
        f"mAP@50={result['map50']:.3f}"
    )

print()
print("-------------------------- BEST THRESHOLD --------------------------")
print("best threshold:", best_result["threshold"])
print("best mAP@50:", f"{best_result['map50']:.3f}")
print("prediction count:", best_result["prediction_count"])
print("saved csv:", RESULT_CSV_PATH)