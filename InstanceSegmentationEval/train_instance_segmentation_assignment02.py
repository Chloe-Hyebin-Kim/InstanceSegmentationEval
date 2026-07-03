import json
import shutil
from pathlib import Path

import cv2
import numpy as np
import torch
from pycocotools import mask as mask_utils
from ultralytics import YOLO


# ============================================================
# 1. 전체 경로 및 학습 설정
# ============================================================
# 이 파일의 목적:
# 1) COCO segmentation 형식의 dataset-coco-seg 데이터를 읽는다.
# 2) YOLO segmentation 학습 형식으로 변환한다.
# 3) YOLO11s-seg 모델을 train set으로 fine-tuning한다.
# 4) 학습된 best.pt 모델을 runs_assignment02 폴더에 저장한다.

# 현재 프로젝트 최상위 폴더 경로
PROJECT_ROOT = Path(r"D:\git\InstanceSegmentationEval\InstanceSegmentationEval")

# 원본 COCO segmentation 데이터셋 폴더
DATASET_ROOT = PROJECT_ROOT / "dataset-coco-seg"

# COCO 데이터셋 split별 이미지 폴더
TRAIN_DIR = DATASET_ROOT / "train"
VALID_DIR = DATASET_ROOT / "valid"
TEST_DIR = DATASET_ROOT / "test"

# COCO annotation JSON 파일 경로
TRAIN_ANN_PATH = TRAIN_DIR / "_annotations.coco.json"
VALID_ANN_PATH = VALID_DIR / "_annotations.coco.json"
TEST_ANN_PATH = TEST_DIR / "_annotations.coco.json"

# YOLO segmentation 형식으로 변환된 데이터셋을 저장할 폴더
YOLO_DATASET_ROOT = PROJECT_ROOT / "dataset-yolo-seg-assignment02"

# YOLO 학습용 이미지/라벨 폴더
# 최종 구조:
# dataset-yolo-seg-assignment02/
#   images/
#     train/
#     val/
#     test/
#   labels/
#     train/
#     val/
#     test/
#   data.yaml
YOLO_IMAGES_ROOT = YOLO_DATASET_ROOT / "images"
YOLO_LABELS_ROOT = YOLO_DATASET_ROOT / "labels"
YOLO_DATA_YAML_PATH = YOLO_DATASET_ROOT / "data.yaml"

# YOLO 학습 결과가 저장될 폴더
RUNS_ROOT = PROJECT_ROOT / "runs_assignment02"
RUN_NAME = "yolo_seg_assignment02"

# 사용할 YOLO segmentation pretrained model 후보
# yolo11s-seg.pt 로드가 실패하면 yolov8s-seg.pt를 대체 모델로 사용한다.
MODEL_CANDIDATES = [
    "yolo11s-seg.pt",
    "yolov8s-seg.pt",
]

# YOLO 학습 hyperparameter
EPOCHS = 100          # 전체 학습 epoch 수
IMGSZ = 640           # 입력 이미지 크기
BATCH = 4             # batch size
PATIENCE = 30         # validation 성능 개선이 없을 때 early stopping patience

# GPU 사용 가능하면 device=0, 아니면 CPU 사용
# RTX 5080 + CUDA PyTorch가 정상 설치되어 있으면 DEVICE는 0이 된다.
DEVICE = 0 if torch.cuda.is_available() else "cpu"


# ============================================================
# 2. 원본 데이터셋 경로 확인
# ============================================================

def check_paths():
    print()
    print("-------------------------- PATH CHECK --------------------------")
    print("PROJECT_ROOT:", PROJECT_ROOT)
    print("DATASET_ROOT:", DATASET_ROOT)
    print("TRAIN_ANN:", TRAIN_ANN_PATH)
    print("VALID_ANN:", VALID_ANN_PATH)
    print("TEST_ANN:", TEST_ANN_PATH)
    print("DEVICE:", DEVICE)
    print()

    required_paths = [
        TRAIN_DIR,
        VALID_DIR,
        TEST_DIR,
        TRAIN_ANN_PATH,
        VALID_ANN_PATH,
        TEST_ANN_PATH,
    ]

    for path in required_paths:
        print(path.exists(), "->", path)

        if not path.exists():
            raise FileNotFoundError(path)


# ============================================================
# 3. COCO segmentation 형식 → YOLO segmentation 형식 변환
# ============================================================
# COCO annotation은 segmentation을 polygon 또는 RLE 형식으로 저장한다.
# 반면 YOLO segmentation label은 한 객체를 한 줄로 저장한다.
#
# YOLO segmentation label 형식:
# class_id x1 y1 x2 y2 x3 y3 ...
#
# 여기서 좌표는 이미지 크기로 나눈 normalized 좌표여야 한다.
# 즉 x는 0~1, y도 0~1 범위로 저장된다.


def make_dirs():
    """
    YOLO segmentation 학습에 필요한 폴더 구조를 만든다.

    생성되는 폴더:
    - images/train
    - images/val
    - images/test
    - labels/train
    - labels/val
    - labels/test
    """

    for split in ["train", "val", "test"]:
        (YOLO_IMAGES_ROOT / split).mkdir(parents=True, exist_ok=True)
        (YOLO_LABELS_ROOT / split).mkdir(parents=True, exist_ok=True)


def normalize_polygon(polygon, width, height):
    """
    COCO polygon 좌표를 YOLO segmentation label에 맞게 정규화한다.

    입력:
    - polygon: COCO polygon 좌표 리스트
      예: [x1, y1, x2, y2, x3, y3, ...]
    - width: 이미지 너비
    - height: 이미지 높이

    처리:
    1) polygon 좌표가 최소 3개 점 이상인지 확인한다.
       segmentation polygon은 최소 3점이 필요하므로 좌표값은 최소 6개여야 한다.
    2) 좌표가 이미지 바깥으로 나가지 않도록 clipping한다.
    3) x는 width로 나누고, y는 height로 나누어 0~1 범위로 변환한다.

    반환:
    - 정상 polygon이면 normalized 좌표 리스트
    - 사용할 수 없는 polygon이면 None
    """

    # polygon은 최소 3개의 점이 필요하다.
    # 점 1개는 x, y 2개 값으로 구성되므로 최소 길이는 6이다.
    if len(polygon) < 6:
        return None

    points = []

    # polygon 리스트는 [x1, y1, x2, y2, ...] 형태이므로 2개씩 읽는다.
    for i in range(0, len(polygon), 2):
        x = float(polygon[i])
        y = float(polygon[i + 1])

        # 이미지 범위를 벗어난 좌표를 이미지 내부로 보정한다.
        x = max(0.0, min(x, width - 1))
        y = max(0.0, min(y, height - 1))

        # YOLO segmentation은 normalized 좌표를 사용한다.
        x_norm = x / width
        y_norm = y / height

        points.extend([x_norm, y_norm])

    if len(points) < 6:
        return None

    return points


def rle_to_largest_polygon(segmentation, height, width):
    """
    COCO RLE segmentation을 polygon으로 변환한다.

    COCO segmentation에는 두 가지 대표 형식이 있다.
    1) polygon format:
       [[x1, y1, x2, y2, ...]]
    2) RLE format:
       {"size": [h, w], "counts": ...}

    YOLO segmentation label은 polygon 좌표가 필요하므로,
    RLE mask를 먼저 binary mask로 decode한 뒤 contour를 찾아 polygon으로 변환한다.

    여러 contour가 발견될 수 있는데, 여기서는 가장 큰 contour만 사용한다.
    이유:
    - YOLO label 한 줄은 하나의 polygon을 표현하는 방식에 가깝다.
    - 가장 큰 contour가 해당 객체의 주 영역일 가능성이 높다.
    """

    # counts가 list이면 uncompressed RLE이므로 pycocotools 형식으로 변환한다.
    if isinstance(segmentation.get("counts"), list):
        rle = mask_utils.frPyObjects(segmentation, height, width)

    # counts가 문자열 또는 bytes인 경우는 encoded RLE이다.
    else:
        rle = segmentation

        # pycocotools.decode는 counts가 bytes 형태인 RLE을 기대할 수 있으므로
        # 문자열이면 bytes로 변환한다.
        if isinstance(rle["counts"], str):
            rle = {
                "size": rle["size"],
                "counts": rle["counts"].encode("utf-8"),
            }

    # RLE을 binary mask로 변환한다.
    mask = mask_utils.decode(rle)

    # mask가 여러 channel로 decode되는 경우 하나의 binary mask로 합친다.
    if mask.ndim == 3:
        mask = np.any(mask, axis=2).astype(np.uint8)
    else:
        mask = mask.astype(np.uint8)

    # binary mask에서 외곽 contour를 찾는다.
    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    # contour가 없으면 polygon으로 변환할 수 없다.
    if len(contours) == 0:
        return None

    # 여러 contour 중 면적이 가장 큰 contour를 객체 polygon으로 사용한다.
    largest_contour = max(contours, key=cv2.contourArea)

    # polygon은 최소 3점이 필요하다.
    if len(largest_contour) < 3:
        return None

    # contour shape을 [x1, y1, x2, y2, ...] 형태로 변환한다.
    polygon = largest_contour.reshape(-1, 2).astype(float).flatten().tolist()

    if len(polygon) < 6:
        return None

    return polygon


def get_annotation_polygon(annotation, image_width, image_height):
    """
    COCO annotation 하나에서 segmentation polygon을 추출한다.

    annotation의 segmentation 형식에 따라 처리 방식이 다르다.

    1) segmentation이 list이면:
       이미 COCO polygon format이므로 사용 가능한 polygon을 골라낸다.

    2) segmentation이 dict이면:
       RLE format이므로 binary mask로 decode한 뒤 polygon으로 변환한다.

    반환:
    - YOLO label로 변환 가능한 polygon 리스트
    - 변환할 수 없으면 None
    """

    segmentation = annotation.get("segmentation", None)

    if segmentation is None:
        return None

    # COCO polygon format
    # 예: [[x1, y1, x2, y2, ...]]
    if isinstance(segmentation, list):
        valid_polygons = []

        for polygon in segmentation:
            if polygon is None:
                continue

            if len(polygon) >= 6:
                valid_polygons.append(polygon)

        if len(valid_polygons) == 0:
            return None

        # 하나의 객체가 여러 polygon으로 나뉘어 있을 수 있다.
        # 여기서는 가장 긴 polygon을 대표 polygon으로 사용한다.
        return max(valid_polygons, key=len)

    # COCO RLE format
    # 예: {"size": [h, w], "counts": "..."}
    if isinstance(segmentation, dict):
        return rle_to_largest_polygon(
            segmentation=segmentation,
            height=image_height,
            width=image_width,
        )

    return None


def convert_one_split(coco_split_name, yolo_split_name, image_dir, annotation_path):
    """
    COCO 데이터셋 split 하나를 YOLO segmentation 형식으로 변환한다.

    예:
    - COCO train → YOLO train
    - COCO valid → YOLO val
    - COCO test  → YOLO test

    처리 과정:
    1) COCO annotation JSON을 읽는다.
    2) image_id 기준으로 annotation을 묶는다.
    3) 각 이미지를 YOLO images 폴더로 복사한다.
    4) 각 이미지에 대응되는 YOLO label txt 파일을 만든다.
    5) annotation polygon을 normalized polygon으로 변환하여 txt에 저장한다.
    """

    print()
    print(f"-------------------------- CONVERT {coco_split_name.upper()} --------------------------")

    # COCO annotation JSON 로드
    with open(annotation_path, "r", encoding="utf-8") as f:
        coco = json.load(f)

    images = coco["images"]
    annotations = coco["annotations"]

    # image_id로 image 정보를 빠르게 찾기 위한 dictionary
    image_id_to_info = {
        int(image["id"]): image
        for image in images
    }

    # image_id별 annotation 목록을 묶는다.
    # 이렇게 하면 이미지 하나를 처리할 때 해당 이미지의 객체들만 빠르게 가져올 수 있다.
    anns_by_image_id = {}

    for ann in annotations:
        image_id = int(ann["image_id"])

        if image_id not in anns_by_image_id:
            anns_by_image_id[image_id] = []

        anns_by_image_id[image_id].append(ann)

    # YOLO 변환 결과가 저장될 이미지/라벨 폴더
    image_output_dir = YOLO_IMAGES_ROOT / yolo_split_name
    label_output_dir = YOLO_LABELS_ROOT / yolo_split_name

    converted_images = 0
    converted_annotations = 0
    skipped_annotations = 0

    # COCO images 목록 기준으로 이미지와 label 파일을 생성한다.
    for image_id, image_info in image_id_to_info.items():
        file_name = image_info["file_name"]
        width = int(image_info["width"])
        height = int(image_info["height"])

        src_image_path = image_dir / file_name

        # annotation에는 있는데 실제 이미지 파일이 없으면 건너뛴다.
        if not src_image_path.exists():
            print("[WARNING] image not found:", src_image_path)
            continue

        # 서로 다른 split 또는 폴더에서 같은 파일명이 있을 수 있으므로
        # image_id를 파일명 앞에 붙여 충돌을 방지한다.
        dst_image_name = f"{image_id}_{Path(file_name).name}"
        dst_image_path = image_output_dir / dst_image_name

        # 원본 이미지를 YOLO images 폴더로 복사한다.
        shutil.copy2(src_image_path, dst_image_path)

        # YOLO label 파일은 이미지 파일명과 같은 stem을 사용하고 확장자는 .txt이다.
        label_path = label_output_dir / f"{Path(dst_image_name).stem}.txt"

        label_lines = []

        # 현재 이미지에 해당하는 모든 annotation을 YOLO label line으로 변환한다.
        for ann in anns_by_image_id.get(image_id, []):
            category_id = int(ann["category_id"])

            # COCO segmentation에서 polygon을 추출한다.
            polygon = get_annotation_polygon(
                annotation=ann,
                image_width=width,
                image_height=height,
            )

            if polygon is None:
                skipped_annotations += 1
                continue

            # polygon 좌표를 YOLO 형식의 normalized 좌표로 변환한다.
            normalized_polygon = normalize_polygon(
                polygon=polygon,
                width=width,
                height=height,
            )

            if normalized_polygon is None:
                skipped_annotations += 1
                continue

            # YOLO segmentation label 한 줄:
            # class_id x1 y1 x2 y2 x3 y3 ...
            values = [str(category_id)]
            values.extend([f"{v:.6f}" for v in normalized_polygon])

            label_lines.append(" ".join(values))
            converted_annotations += 1

        # 이미지 하나당 label txt 파일 하나를 생성한다.
        # 객체가 없는 이미지라도 빈 txt 파일을 만들 수 있다.
        with open(label_path, "w", encoding="utf-8") as f:
            if len(label_lines) > 0:
                f.write("\n".join(label_lines))

        converted_images += 1

    print("converted_images:", converted_images)
    print("converted_annotations:", converted_annotations)
    print("skipped_annotations:", skipped_annotations)


def write_data_yaml():
    """
    Ultralytics YOLO 학습에 필요한 data.yaml 파일을 생성한다.

    data.yaml에는 다음 정보가 들어간다.
    - YOLO dataset root path
    - train image path
    - validation image path
    - test image path
    - class id와 class name 목록

    YOLO 학습 시 model.train(data=...)에 이 yaml 파일을 넘긴다.
    """

    # class 목록은 train annotation의 categories를 기준으로 사용한다.
    with open(TRAIN_ANN_PATH, "r", encoding="utf-8") as f:
        coco = json.load(f)

    # category id 순서대로 정렬한다.
    categories = sorted(coco["categories"], key=lambda x: int(x["id"]))

    lines = []
    lines.append(f"path: {YOLO_DATASET_ROOT.as_posix()}")
    lines.append("train: images/train")
    lines.append("val: images/val")
    lines.append("test: images/test")
    lines.append("")
    lines.append("names:")

    # COCO category id를 그대로 YOLO class id로 사용한다.
    # 이렇게 하면 학습 후 예측 class_id와 COCO category_id를 다시 매핑하기 쉽다.
    for category in categories:
        category_id = int(category["id"])
        category_name = category["name"]
        lines.append(f"  {category_id}: {category_name}")

    with open(YOLO_DATA_YAML_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print()
    print("-------------------------- DATA YAML --------------------------")
    print("saved:", YOLO_DATA_YAML_PATH)
    print("\n".join(lines))


def convert_coco_to_yolo():
    """
    전체 COCO dataset을 YOLO segmentation dataset으로 변환한다.

    실행 순서:
    1) YOLO용 폴더 생성
    2) train split 변환
    3) valid split을 YOLO의 val split으로 변환
    4) test split 변환
    5) data.yaml 생성
    """

    make_dirs()

    convert_one_split(
        coco_split_name="train",
        yolo_split_name="train",
        image_dir=TRAIN_DIR,
        annotation_path=TRAIN_ANN_PATH,
    )

    convert_one_split(
        coco_split_name="valid",
        yolo_split_name="val",
        image_dir=VALID_DIR,
        annotation_path=VALID_ANN_PATH,
    )

    convert_one_split(
        coco_split_name="test",
        yolo_split_name="test",
        image_dir=TEST_DIR,
        annotation_path=TEST_ANN_PATH,
    )

    write_data_yaml()


# ============================================================
# 4. YOLO segmentation 모델 학습
# ============================================================

def load_yolo_model():
    """
    YOLO segmentation pretrained model을 로드한다.

    우선 yolo11s-seg.pt를 시도한다.
    만약 다운로드 또는 로드가 실패하면 yolov8s-seg.pt를 시도한다.

    이렇게 fallback 구조를 둔 이유:
    - 환경에 따라 특정 weight 다운로드가 실패할 수 있다.
    - 그래도 학습 실험을 계속 진행할 수 있게 하기 위해서이다.
    """

    last_error = None

    for model_name in MODEL_CANDIDATES:
        try:
            print()
            print("-------------------------- MODEL INIT --------------------------")
            print("try model:", model_name)

            # Ultralytics YOLO 객체 생성
            # pretrained weight가 없으면 자동 다운로드를 시도한다.
            model = YOLO(model_name)

            print("loaded model:", model_name)
            return model

        except Exception as e:
            print("[WARNING] failed:", model_name)
            print(e)
            last_error = e

    # 모든 후보 모델 로드에 실패하면 학습을 중단한다.
    raise RuntimeError(f"YOLO model load failed. last_error={last_error}")


def train_yolo_segmentation():
    """
    YOLO segmentation 모델을 fine-tuning한다.

    학습 입력:
    - data: COCO에서 YOLO 형식으로 변환한 data.yaml
    - task: segment
    - epochs: 100
    - imgsz: 640
    - batch: 4
    - device: GPU가 있으면 0, 없으면 CPU

    학습 결과:
    - runs_assignment02/yolo_seg_assignment02/weights/best.pt
    - runs_assignment02/yolo_seg_assignment02/weights/last.pt

    best.pt는 validation 성능이 가장 좋았던 checkpoint이며,
    이후 evaluate_trained_model_assignment02.py에서 최종 평가에 사용한다.
    """

    print()
    print("-------------------------- TRAIN YOLO SEGMENTATION --------------------------")
    print("data:", YOLO_DATA_YAML_PATH)
    print("epochs:", EPOCHS)
    print("imgsz:", IMGSZ)
    print("batch:", BATCH)
    print("device:", DEVICE)

    # pretrained YOLO segmentation model 로드
    model = load_yolo_model()

    # YOLO segmentation fine-tuning 실행
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

    # 학습 완료 후 best.pt 경로 확인
    best_model_path = RUNS_ROOT / RUN_NAME / "weights" / "best.pt"

    print()
    print("-------------------------- TRAIN DONE --------------------------")
    print("best model:", best_model_path)
    print("exists:", best_model_path.exists())


# ============================================================
# 5. main 실행부
# ============================================================

def main():
    """
    전체 실행 흐름을 관리하는 함수이다.

    실행 순서:
    1) 원본 COCO dataset 경로 확인
    2) COCO segmentation dataset을 YOLO segmentation format으로 변환
    3) YOLO11s-seg 모델 fine-tuning
    """

    check_paths()
    convert_coco_to_yolo()
    train_yolo_segmentation()


if __name__ == "__main__":
    main()