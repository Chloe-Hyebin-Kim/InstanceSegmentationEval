
from ultralytics import YOLO

from config import (
    TRAINED_YOLO_MODEL_PATH,
    VALID_DIR,
    TEST_DIR,
    VALID_ANN_PATH,
    TEST_ANN_PATH,
    VALID_YOLO_PRED_PATH,
    TEST_YOLO_PRED_PATH,
    VALID_YOLO_CLASS_METRIC_PATH,
    TEST_YOLO_CLASS_METRIC_PATH,
    YOLO_SUMMARY_PATH,
    BASELINE_VALID_MAP50,
    BASELINE_TEST_MAP50,
    DEVICE,
)
from evaluation import evaluate_split
from inference import run_yolo_inference_to_coco_json
from utils import ensure_result_dir


# ============================================================
# 1. baseline vs YOLO summary 저장
# ============================================================

def save_yolo_summary(valid_map50, test_map50):
    """
    01번 baseline RF-DETR 모델과 02번 fine-tuned YOLO 모델의 성능을 비교한다.
    test mAP@50이 baseline보다 높으면 과제 02번 목표 달성으로 판단한다.
    """

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

    with open(YOLO_SUMMARY_PATH, "w", encoding="utf-8-sig") as f:
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
    print("summary saved:", YOLO_SUMMARY_PATH)

    print()

    if test_map50 > BASELINE_TEST_MAP50:
        print("RESULT: SUCCESS - YOLO model is better than baseline on test mAP@50.")
    else:
        print("RESULT: NOT YET - YOLO model did not exceed baseline test mAP@50.")


# ============================================================
# 2. main 실행부
# ============================================================

def main():
    """
    02번 평가 실행 흐름이다.

    실행 순서:
    1) 학습된 YOLO best.pt 존재 여부 확인
    2) valid/test set YOLO 추론
    3) YOLO prediction을 COCO result JSON으로 저장
    4) COCOeval mAP@50 및 class metric 계산
    5) baseline RF-DETR 성능과 비교
    """

    ensure_result_dir()

    print()
    print("-------------------------- EVALUATE TRAINED YOLO MODEL --------------------------")
    print("model:", TRAINED_YOLO_MODEL_PATH)
    print("exists:", TRAINED_YOLO_MODEL_PATH.exists())
    print("device:", DEVICE)

    if not TRAINED_YOLO_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"trained model not found: {TRAINED_YOLO_MODEL_PATH}\n"
            f"먼저 02_train_yolo_segmentation.py를 실행하세요."
        )

    model = YOLO(str(TRAINED_YOLO_MODEL_PATH))

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

    save_yolo_summary(
        valid_map50=valid_map50,
        test_map50=test_map50,
    )


if __name__ == "__main__":
    main()
