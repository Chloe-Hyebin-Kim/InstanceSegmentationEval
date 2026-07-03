
import json
import shutil
from pathlib import Path

import cv2
import numpy as np
from pycocotools import mask as mask_utils

from config import (
    PROJECT_ROOT,
    MODEL_PATH,
    TRAIN_DIR,
    VALID_DIR,
    TEST_DIR,
    TRAIN_ANN_PATH,
    VALID_ANN_PATH,
    TEST_ANN_PATH,
    RESULT_DIR,
    YOLO_IMAGES_ROOT,
    YOLO_LABELS_ROOT,
    YOLO_DATASET_ROOT,
    YOLO_DATA_YAML_PATH,
)


# ============================================================
# 1. 공통 파일/폴더 유틸리티
# ============================================================

def ensure_result_dir():
    """
    results 폴더가 없으면 생성한다.
    모든 prediction json, class metric csv, summary csv는 이 폴더에 저장된다.
    """

    RESULT_DIR.mkdir(parents=True, exist_ok=True)


def load_json(path):
    """
    JSON 파일을 UTF-8로 읽어서 Python 객체로 반환한다.
    """

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data, path):
    """
    Python 객체를 JSON 파일로 저장한다.
    prediction 결과 저장에 사용한다.
    """

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f)


def check_common_paths(include_train=False):
    """
    실행 전에 필요한 주요 파일/폴더가 존재하는지 확인한다.

    include_train=False:
    - 01번 baseline 평가처럼 valid/test만 필요한 경우

    include_train=True:
    - 02번 YOLO 학습처럼 train/valid/test가 모두 필요한 경우
    """

    print()
    print("-------------------------- PATH CHECK --------------------------")
    print("PROJECT_ROOT:", PROJECT_ROOT)
    print("MODEL_PATH:", MODEL_PATH)
    print("DATASET_ROOT:", PROJECT_ROOT / "dataset-coco-seg")
    print("DEVICE check is handled in config.py")
    print()

    required_paths = [
        MODEL_PATH,
        VALID_DIR,
        TEST_DIR,
        VALID_ANN_PATH,
        TEST_ANN_PATH,
    ]

    if include_train:
        required_paths.extend([
            TRAIN_DIR,
            TRAIN_ANN_PATH,
        ])

    for path in required_paths:
        print(path.exists(), "->", path)

        if not path.exists():
            raise FileNotFoundError(path)

    ensure_result_dir()
    print("PATH CHECK OK")


def print_dataset_categories(annotation_path, title):
    """
    COCO annotation JSON 안에 정의된 category id와 class name을 출력한다.
    모델 예측 class_id와 COCO category_id가 맞는지 확인하기 위해 사용한다.
    """

    coco = load_json(annotation_path)

    print()
    print(f"-------------------------- {title} CATEGORIES --------------------------")

    for category in sorted(coco["categories"], key=lambda x: int(x["id"])):
        print(f'id={category["id"]}, name={category["name"]}')


# ============================================================
# 2. 깨진 COCO JSON metadata 보정
# ============================================================

def fix_coco_json_file(annotation_path):
    """
    일부 train annotation JSON의 info metadata에 key 없이 문자열이 들어가 있어
    json.decoder.JSONDecodeError가 발생할 수 있다.

    예시 오류 형태:
    "version":"4","RebuilderAI bottom-subset","contributor":""

    정상 형태:
    "version":"4","description":"RebuilderAI bottom-subset","contributor":""

    실제 annotation, image, category 정보는 건드리지 않고 metadata만 수정한다.
    """

    annotation_path = Path(annotation_path)

    print()
    print("-------------------------- CHECK JSON --------------------------")
    print("path:", annotation_path)

    text = annotation_path.read_text(encoding="utf-8", errors="replace")

    try:
        data = json.loads(text)
        print("JSON already OK")
        print("images:", len(data.get("images", [])))
        print("annotations:", len(data.get("annotations", [])))
        print("categories:", len(data.get("categories", [])))
        return

    except json.JSONDecodeError as e:
        print("JSON ERROR detected")
        print("message:", e)
        print("line:", e.lineno)
        print("column:", e.colno)
        print("position:", e.pos)

    broken_text = '"version":"4","RebuilderAI bottom-subset","contributor"'
    fixed_text = '"version":"4","description":"RebuilderAI bottom-subset","contributor"'

    if broken_text not in text:
        print()
        print("known broken pattern을 찾지 못했습니다.")
        print("파일 앞부분:")
        print(text[:300])
        raise ValueError("자동 수정할 수 없는 JSON 오류입니다.")

    backup_path = annotation_path.with_name("_annotations.coco.backup.json")

    if not backup_path.exists():
        backup_path.write_text(text, encoding="utf-8")
        print("backup saved:", backup_path)
    else:
        print("backup already exists:", backup_path)

    fixed = text.replace(broken_text, fixed_text, 1)
    data = json.loads(fixed)
    annotation_path.write_text(fixed, encoding="utf-8")

    print("JSON fixed successfully")
    print("images:", len(data.get("images", [])))
    print("annotations:", len(data.get("annotations", [])))
    print("categories:", len(data.get("categories", [])))
    print("fixed file saved:", annotation_path)


def fix_all_coco_json_files():
    """
    train/valid/test annotation JSON을 모두 검사한다.
    깨진 파일은 수정하고, 이미 정상인 파일은 그대로 둔다.
    """

    for annotation_path in [TRAIN_ANN_PATH, VALID_ANN_PATH, TEST_ANN_PATH]:
        fix_coco_json_file(annotation_path)


# ============================================================
# 3. mask encoding 유틸리티
# ============================================================

def encode_binary_mask(binary_mask):
    """
    binary mask를 COCO RLE 형식으로 변환한다.

    COCOeval에서 segmentation 평가를 하려면 prediction의 segmentation 필드가
    COCO가 이해할 수 있는 RLE 또는 polygon 형식이어야 한다.
    여기서는 binary mask를 RLE로 저장한다.
    """

    binary_mask = np.asfortranarray(binary_mask.astype(np.uint8))
    rle = mask_utils.encode(binary_mask)
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle


# ============================================================
# 4. COCO segmentation → YOLO segmentation 변환 유틸리티
# ============================================================

def make_yolo_dirs():
    """
    YOLO segmentation 학습에 필요한 폴더 구조를 만든다.
    """

    for split in ["train", "val", "test"]:
        (YOLO_IMAGES_ROOT / split).mkdir(parents=True, exist_ok=True)
        (YOLO_LABELS_ROOT / split).mkdir(parents=True, exist_ok=True)


def normalize_polygon(polygon, width, height):
    """
    COCO polygon 좌표를 YOLO segmentation label에 맞게 0~1 범위로 정규화한다.
    """

    if len(polygon) < 6:
        return None

    points = []

    for i in range(0, len(polygon), 2):
        x = float(polygon[i])
        y = float(polygon[i + 1])

        x = max(0.0, min(x, width - 1))
        y = max(0.0, min(y, height - 1))

        points.extend([x / width, y / height])

    if len(points) < 6:
        return None

    return points


def rle_to_largest_polygon(segmentation, height, width):
    """
    COCO RLE segmentation을 binary mask로 decode한 뒤 가장 큰 contour를 polygon으로 변환한다.
    """

    if isinstance(segmentation.get("counts"), list):
        rle = mask_utils.frPyObjects(segmentation, height, width)
    else:
        rle = segmentation

        if isinstance(rle["counts"], str):
            rle = {
                "size": rle["size"],
                "counts": rle["counts"].encode("utf-8"),
            }

    mask = mask_utils.decode(rle)

    if mask.ndim == 3:
        mask = np.any(mask, axis=2).astype(np.uint8)
    else:
        mask = mask.astype(np.uint8)

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    if len(contours) == 0:
        return None

    largest_contour = max(contours, key=cv2.contourArea)

    if len(largest_contour) < 3:
        return None

    polygon = largest_contour.reshape(-1, 2).astype(float).flatten().tolist()

    if len(polygon) < 6:
        return None

    return polygon


def get_annotation_polygon(annotation, image_width, image_height):
    """
    COCO annotation 하나에서 YOLO label로 변환 가능한 polygon을 추출한다.
    polygon format이면 가장 긴 polygon을 사용하고, RLE format이면 polygon으로 변환한다.
    """

    segmentation = annotation.get("segmentation", None)

    if segmentation is None:
        return None

    if isinstance(segmentation, list):
        valid_polygons = [polygon for polygon in segmentation if polygon is not None and len(polygon) >= 6]

        if len(valid_polygons) == 0:
            return None

        return max(valid_polygons, key=len)

    if isinstance(segmentation, dict):
        return rle_to_largest_polygon(
            segmentation=segmentation,
            height=image_height,
            width=image_width,
        )

    return None


def convert_one_split(coco_split_name, yolo_split_name, image_dir, annotation_path):
    """
    COCO split 하나를 YOLO segmentation split 하나로 변환한다.
    예: COCO valid → YOLO val
    """

    print()
    print(f"-------------------------- CONVERT {coco_split_name.upper()} --------------------------")

    coco = load_json(annotation_path)

    images = coco["images"]
    annotations = coco["annotations"]

    image_id_to_info = {int(image["id"]): image for image in images}

    anns_by_image_id = {}

    for ann in annotations:
        image_id = int(ann["image_id"])
        anns_by_image_id.setdefault(image_id, []).append(ann)

    image_output_dir = YOLO_IMAGES_ROOT / yolo_split_name
    label_output_dir = YOLO_LABELS_ROOT / yolo_split_name

    converted_images = 0
    converted_annotations = 0
    skipped_annotations = 0

    for image_id, image_info in image_id_to_info.items():
        file_name = image_info["file_name"]
        width = int(image_info["width"])
        height = int(image_info["height"])

        src_image_path = image_dir / file_name

        if not src_image_path.exists():
            print("[WARNING] image not found:", src_image_path)
            continue

        dst_image_name = f"{image_id}_{Path(file_name).name}"
        dst_image_path = image_output_dir / dst_image_name

        shutil.copy2(src_image_path, dst_image_path)

        label_path = label_output_dir / f"{Path(dst_image_name).stem}.txt"
        label_lines = []

        for ann in anns_by_image_id.get(image_id, []):
            category_id = int(ann["category_id"])

            polygon = get_annotation_polygon(
                annotation=ann,
                image_width=width,
                image_height=height,
            )

            if polygon is None:
                skipped_annotations += 1
                continue

            normalized_polygon = normalize_polygon(
                polygon=polygon,
                width=width,
                height=height,
            )

            if normalized_polygon is None:
                skipped_annotations += 1
                continue

            values = [str(category_id)]
            values.extend([f"{v:.6f}" for v in normalized_polygon])

            label_lines.append(" ".join(values))
            converted_annotations += 1

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
    class id는 COCO category_id를 그대로 사용한다.
    """

    coco = load_json(TRAIN_ANN_PATH)
    categories = sorted(coco["categories"], key=lambda x: int(x["id"]))

    lines = []
    lines.append(f"path: {YOLO_DATASET_ROOT.as_posix()}")
    lines.append("train: images/train")
    lines.append("val: images/val")
    lines.append("test: images/test")
    lines.append("")
    lines.append("names:")

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
    train/valid/test 전체 COCO segmentation dataset을 YOLO segmentation format으로 변환한다.
    """

    make_yolo_dirs()

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
