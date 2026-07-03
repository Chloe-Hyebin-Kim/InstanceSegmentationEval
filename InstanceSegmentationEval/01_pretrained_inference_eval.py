
from rfdetr import RFDETRSegMedium

from config import (
    MODEL_PATH,
    VALID_DIR,
    TEST_DIR,
    VALID_ANN_PATH,
    TEST_ANN_PATH,
    VALID_BASELINE_PRED_PATH,
    TEST_BASELINE_PRED_PATH,
    VALID_BASELINE_CLASS_METRIC_PATH,
    TEST_BASELINE_CLASS_METRIC_PATH,
    BASELINE_SUMMARY_PATH,
)
from evaluation import evaluate_split
from inference import run_rfdetr_inference_to_coco_json
from utils import check_common_paths, print_dataset_categories


# ============================================================
# 1. baseline summary 저장
# ============================================================

def save_baseline_summary(valid_map50, test_map50):
    """
    제공된 pretrained RF-DETR 모델의 valid/test mAP@50 결과를 CSV로 저장한다.
    """

    with open(BASELINE_SUMMARY_PATH, "w", encoding="utf-8-sig") as f:
        f.write("model,valid_map50,test_map50\n")
        f.write(f"IS_pretrained_bottom,{valid_map50},{test_map50}\n")

    print()
    print("-------------------------- FINAL SUMMARY --------------------------")
    print("VALID mAP@50:", f"{valid_map50:.6f}")
    print("TEST  mAP@50:", f"{test_map50:.6f}")
    print("summary saved:", BASELINE_SUMMARY_PATH)


# ============================================================
# 2. main 실행부
# ============================================================

def main():
    """
    01번 과제 실행 흐름이다.

    목적:
    - 제공된 IS_pretrained_bottom.pt 모델을 valid/test set에서 평가한다.
    - COCOeval segmentation mAP@50을 계산한다.
    - class별 AP, Precision, Recall을 CSV로 저장한다.

    실행 순서:
    1) 경로 확인
    2) valid/test category 확인
    3) pretrained RF-DETR 모델 로드
    4) valid/test 추론 결과를 COCO prediction JSON으로 저장
    5) valid/test mAP@50 및 class metric 계산
    6) 최종 summary 저장
    """

    check_common_paths(include_train=False)

    print_dataset_categories(VALID_ANN_PATH, "VALID")
    print_dataset_categories(TEST_ANN_PATH, "TEST")

    print()
    print("-------------------------- MODEL LOAD --------------------------")
    model = RFDETRSegMedium.from_checkpoint(str(MODEL_PATH))

    print("model loaded:", MODEL_PATH)
    print("model.class_names:")
    print(model.class_names)

    run_rfdetr_inference_to_coco_json(
        model=model,
        image_dir=VALID_DIR,
        annotation_path=VALID_ANN_PATH,
        output_path=VALID_BASELINE_PRED_PATH,
        title="VALID",
    )

    run_rfdetr_inference_to_coco_json(
        model=model,
        image_dir=TEST_DIR,
        annotation_path=TEST_ANN_PATH,
        output_path=TEST_BASELINE_PRED_PATH,
        title="TEST",
    )

    valid_map50 = evaluate_split(
        annotation_path=VALID_ANN_PATH,
        prediction_path=VALID_BASELINE_PRED_PATH,
        class_metric_path=VALID_BASELINE_CLASS_METRIC_PATH,
        title="VALID",
    )

    test_map50 = evaluate_split(
        annotation_path=TEST_ANN_PATH,
        prediction_path=TEST_BASELINE_PRED_PATH,
        class_metric_path=TEST_BASELINE_CLASS_METRIC_PATH,
        title="TEST",
    )

    save_baseline_summary(valid_map50, test_map50)


if __name__ == "__main__":
    main()
