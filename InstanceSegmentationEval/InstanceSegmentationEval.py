import csv

from rfdetr import RFDETRSegMedium

from pretrained_inference_eval_assignment01 import (
    MODEL_PATH,
    VALID_DIR,
    TEST_DIR,
    VALID_ANN_PATH,
    TEST_ANN_PATH,
    VALID_PRED_PATH,
    TEST_PRED_PATH,
    VALID_CLASS_METRIC_PATH,
    TEST_CLASS_METRIC_PATH,
    SUMMARY_PATH,
    CONF_THRES,
    IOU_THRES,
    check_paths,
    print_dataset_categories,
    run_inference,
    evaluate_split,
)


# ============================================================
# main
# ============================================================

def main():
    check_paths()

    print_dataset_categories(VALID_ANN_PATH, "VALID")
    print_dataset_categories(TEST_ANN_PATH, "TEST")

    print()
    print("-------------------------- MODEL LOAD --------------------------")

    model = RFDETRSegMedium.from_checkpoint(str(MODEL_PATH))

    print("model loaded:", MODEL_PATH)

    model_class_names = getattr(model, "class_names", None)

    print("model.class_names:")
    print(model_class_names)

    run_inference(
        model=model,
        image_dir=VALID_DIR,
        annotation_path=VALID_ANN_PATH,
        output_path=VALID_PRED_PATH,
        title="VALID",
    )

    run_inference(
        model=model,
        image_dir=TEST_DIR,
        annotation_path=TEST_ANN_PATH,
        output_path=TEST_PRED_PATH,
        title="TEST",
    )

    valid_map50 = evaluate_split(
        annotation_path=VALID_ANN_PATH,
        prediction_path=VALID_PRED_PATH,
        class_metric_path=VALID_CLASS_METRIC_PATH,
        title="VALID",
    )

    test_map50 = evaluate_split(
        annotation_path=TEST_ANN_PATH,
        prediction_path=TEST_PRED_PATH,
        class_metric_path=TEST_CLASS_METRIC_PATH,
        title="TEST",
    )

    with open(SUMMARY_PATH, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "split",
                "map50",
                "conf_threshold",
                "iou_threshold",
            ],
        )

        writer.writeheader()

        writer.writerow({
            "split": "valid",
            "map50": valid_map50,
            "conf_threshold": CONF_THRES,
            "iou_threshold": IOU_THRES,
        })

        writer.writerow({
            "split": "test",
            "map50": test_map50,
            "conf_threshold": CONF_THRES,
            "iou_threshold": IOU_THRES,
        })

    print()
    print("-------------------------- FINAL SUMMARY --------------------------")
    print(f"VALID mAP@50: {valid_map50:.6f}")
    print(f"TEST  mAP@50: {test_map50:.6f}")
    print(f"summary saved: {SUMMARY_PATH}")


if __name__ == "__main__":
    main()