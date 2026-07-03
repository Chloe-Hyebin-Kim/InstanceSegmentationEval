
from ultralytics import YOLO

from config import (
    YOLO_DATA_YAML_PATH,
    RUNS_ROOT,
    RUN_NAME,
    MODEL_CANDIDATES,
    EPOCHS,
    IMGSZ,
    BATCH,
    PATIENCE,
    DEVICE,
)
from utils import (
    check_common_paths,
    fix_all_coco_json_files,
    convert_coco_to_yolo,
)


# ============================================================
# 1. YOLO pretrained model 로드
# ============================================================

def load_yolo_model():
    """
    YOLO segmentation pretrained model을 로드한다.

    우선 yolo11s-seg.pt를 사용하고,
    로드 실패 시 yolov8s-seg.pt를 대체 모델로 사용한다.
    """

    last_error = None

    for model_name in MODEL_CANDIDATES:
        try:
            print()
            print("-------------------------- MODEL INIT --------------------------")
            print("try model:", model_name)

            model = YOLO(model_name)

            print("loaded model:", model_name)
            return model

        except Exception as e:
            print("[WARNING] failed:", model_name)
            print(e)
            last_error = e

    raise RuntimeError(f"YOLO model load failed. last_error={last_error}")


# ============================================================
# 2. YOLO segmentation fine-tuning
# ============================================================

def train_yolo_segmentation():
    """
    COCO에서 YOLO format으로 변환한 dataset-yolo-seg-assignment02를 사용하여
    YOLO segmentation 모델을 fine-tuning한다.

    학습 결과:
    - runs_assignment02/yolo_seg_assignment02/weights/best.pt
    - runs_assignment02/yolo_seg_assignment02/weights/last.pt
    """

    print()
    print("-------------------------- TRAIN YOLO SEGMENTATION --------------------------")
    print("data:", YOLO_DATA_YAML_PATH)
    print("epochs:", EPOCHS)
    print("imgsz:", IMGSZ)
    print("batch:", BATCH)
    print("device:", DEVICE)

    model = load_yolo_model()

    model.train(
        data=str(YOLO_DATA_YAML_PATH),
        task="segment",
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        patience=PATIENCE,
        device=DEVICE,
        project=str(RUNS_ROOT),
        name=RUN_NAME,
        exist_ok=True,
    )

    best_model_path = RUNS_ROOT / RUN_NAME / "weights" / "best.pt"

    print()
    print("-------------------------- TRAIN DONE --------------------------")
    print("best model:", best_model_path)
    print("exists:", best_model_path.exists())


# ============================================================
# 3. main 실행부
# ============================================================

def main():
    """
    02번 학습 실행 흐름이다.

    실행 순서:
    1) train/valid/test 경로 확인
    2) COCO JSON metadata 오류 보정
    3) COCO segmentation dataset을 YOLO segmentation format으로 변환
    4) YOLO11s-seg 모델 fine-tuning
    """

    check_common_paths(include_train=True)
    fix_all_coco_json_files()
    convert_coco_to_yolo()
    train_yolo_segmentation()


if __name__ == "__main__":
    main()
