
from pathlib import Path

import torch


# ============================================================
# 1. 프로젝트 기본 경로 설정
# ============================================================
# 이 프로젝트의 모든 코드는 프로젝트 루트 폴더에서 실행한다고 가정한다.
# 이 파일(config.py)이 D:\git\InstanceSegmentationEval\InstanceSegmentationEval 안에 있으면
# PROJECT_ROOT도 자동으로 그 폴더를 가리킨다.

PROJECT_ROOT = Path(__file__).resolve().parent

# 제공된 사전학습 RF-DETR segmentation 모델 파일
MODEL_PATH = PROJECT_ROOT / "IS_pretrained_bottom.pt"

# 원본 COCO segmentation 데이터셋 경로
DATASET_ROOT = PROJECT_ROOT / "dataset-coco-seg"

TRAIN_DIR = DATASET_ROOT / "train"
VALID_DIR = DATASET_ROOT / "valid"
TEST_DIR = DATASET_ROOT / "test"

TRAIN_ANN_PATH = TRAIN_DIR / "_annotations.coco.json"
VALID_ANN_PATH = VALID_DIR / "_annotations.coco.json"
TEST_ANN_PATH = TEST_DIR / "_annotations.coco.json"

# 결과 저장 폴더
RESULT_DIR = PROJECT_ROOT / "results"


# ============================================================
# 2. 01번 baseline RF-DETR 평가 결과 파일 경로
# ============================================================
# 01_pretrained_inference_eval.py에서 생성되는 파일들이다.

VALID_BASELINE_PRED_PATH = RESULT_DIR / "valid_predictions.json"
TEST_BASELINE_PRED_PATH = RESULT_DIR / "test_predictions.json"

VALID_BASELINE_CLASS_METRIC_PATH = RESULT_DIR / "valid_class_metrics.csv"
TEST_BASELINE_CLASS_METRIC_PATH = RESULT_DIR / "test_class_metrics.csv"

BASELINE_SUMMARY_PATH = RESULT_DIR / "summary_map50.csv"


# ============================================================
# 3. 02번 YOLO 학습/평가 관련 경로
# ============================================================
# COCO segmentation dataset을 YOLO segmentation format으로 변환한 결과를 저장한다.

YOLO_DATASET_ROOT = PROJECT_ROOT / "dataset-yolo-seg-assignment02"
YOLO_IMAGES_ROOT = YOLO_DATASET_ROOT / "images"
YOLO_LABELS_ROOT = YOLO_DATASET_ROOT / "labels"
YOLO_DATA_YAML_PATH = YOLO_DATASET_ROOT / "data.yaml"

# YOLO 학습 결과 저장 위치
RUNS_ROOT = PROJECT_ROOT / "runs_assignment02"
RUN_NAME = "yolo_seg_assignment02"

# YOLO 학습 후 생성되는 best model
TRAINED_YOLO_MODEL_PATH = RUNS_ROOT / RUN_NAME / "weights" / "best.pt"

# YOLO valid/test 예측 결과 저장 경로
VALID_YOLO_PRED_PATH = RESULT_DIR / "valid_yolo_assignment02_predictions.json"
TEST_YOLO_PRED_PATH = RESULT_DIR / "test_yolo_assignment02_predictions.json"

# YOLO class별 평가 결과 저장 경로
VALID_YOLO_CLASS_METRIC_PATH = RESULT_DIR / "valid_yolo_assignment02_class_metrics.csv"
TEST_YOLO_CLASS_METRIC_PATH = RESULT_DIR / "test_yolo_assignment02_class_metrics.csv"

# baseline과 YOLO 최종 비교 결과 저장 경로
YOLO_SUMMARY_PATH = RESULT_DIR / "summary_assignment02_yolo.csv"


# ============================================================
# 4. 학습 및 평가 hyperparameter
# ============================================================
# RF-DETR baseline 추론 confidence threshold
RFDETR_CONF_THRES = 0.1

# YOLO COCOeval 평가용 confidence threshold
# COCO AP 계산에서는 confidence score 순위 전체가 중요하므로 낮게 설정한다.
YOLO_CONF_THRES = 0.001

# NMS 및 matching에 사용할 IoU threshold
IOU_THRES = 0.5

# YOLO 입력 이미지 크기
IMGSZ = 640

# YOLO 학습 설정
EPOCHS = 100
BATCH = 4
PATIENCE = 30

# GPU가 사용 가능하면 GPU 0번 사용, 아니면 CPU 사용
DEVICE = 0 if torch.cuda.is_available() else "cpu"


# ============================================================
# 5. baseline 성능 값
# ============================================================
# 01번 RF-DETR baseline 평가에서 얻은 결과이다.
# 02번 YOLO 모델이 이 test mAP@50보다 높으면 성능 개선 성공으로 판단한다.

BASELINE_VALID_MAP50 = 0.532781
BASELINE_TEST_MAP50 = 0.567696


# ============================================================
# 6. YOLO pretrained model 후보
# ============================================================
# yolo11s-seg.pt 로드가 실패하면 yolov8s-seg.pt로 fallback한다.

MODEL_CANDIDATES = [
    "yolo11s-seg.pt",
    "yolov8s-seg.pt",
]
