import json
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from pycocotools import mask as mask_utils
from ultralytics import YOLO

from pretrained_inference_eval_assignment01 import (
    PROJECT_ROOT,
    VALID_DIR,
    TEST_DIR,
    VALID_ANN_PATH,
    TEST_ANN_PATH,
    RESULT_DIR,
    evaluate_split,
)


# ============================================================
# 1. 전체 경로 및 평가 설정
# ============================================================
# 이 파일의 목적:
# 1) train_instance_segmentation_assignment02.py로 학습한 YOLO best.pt 모델을 불러온다.
# 2) valid/test set에 대해 YOLO segmentation 추론을 수행한다.
# 3) YOLO 예측 결과를 COCOeval이 읽을 수 있는 COCO result JSON 형식으로 저장한다.
# 4) 기존 pretrained RF-DETR 모델과 동일한 COCOeval 방식으로 mAP@50을 계산한다.
# 5) baseline 모델과 YOLO fine-tuning 모델의 성능을 비교하여 summary csv로 저장한다.


# YOLO 학습 결과가 저장된 폴더
RUNS_ROOT = PROJECT_ROOT / "runs_assignment02"
RUN_NAME = "yolo_seg_assignment02"

# 학습된 YOLO 모델 checkpoint
# train_instance_segmentation_assignment02.py 실행 후 생성되는 best.pt를 사용한다.
TRAINED_MODEL_PATH = RUNS_ROOT / RUN_NAME / "weights" / "best.pt"

# YOLO valid/test 예측 결과를 COCO result JSON 형식으로 저장할 경로
VALID_YOLO_PRED_PATH = RESULT_DIR / "valid_yolo_assignment02_predictions.json"
TEST_YOLO_PRED_PATH = RESULT_DIR / "test_yolo_assignment02_predictions.json"

# class별 AP, TP, FP, FN, Precision, Recall 결과를 저장할 CSV 경로
VALID_YOLO_CLASS_METRIC_PATH = RESULT_DIR / "valid_yolo_assignment02_class_metrics.csv"
TEST_YOLO_CLASS_METRIC_PATH = RESULT_DIR / "test_yolo_assignment02_class_metrics.csv"

# baseline 모델과 YOLO 모델의 최종 mAP@50 비교 결과를 저장할 CSV 경로
SUMMARY_PATH = RESULT_DIR / "summary_assignment02_yolo.csv"

# 1-1에서 측정한 기존 pretrained RF-DETR 모델의 baseline 성능
# 이 값과 YOLO fine-tuning 모델의 valid/test mAP@50을 비교한다.
BASELINE_VALID_MAP50 = 0.532781
BASELINE_TEST_MAP50 = 0.567696

# YOLO 추론 설정
# CONF_THRES는 COCOeval AP 계산을 위해 낮게 설정한다.
# AP는 confidence score 순위 전체를 사용하므로, 너무 높은 threshold를 주면
# 낮은 confidence이지만 정답일 수 있는 후보들이 제거되어 AP가 낮아질 수 있다.
CONF_THRES = 0.001

# NMS IoU threshold
IOU_THRES = 0.5

# YOLO 추론 이미지 크기
IMGSZ = 640

# CUDA 사용 가능하면 GPU 0번 사용, 아니면 CPU 사용
DEVICE = 0 if torch.cuda.is_available() else "cpu"


# ============================================================
# 2. binary mask → COCO RLE encoding
# ============================================================

def encode_binary_mask(binary_mask):
    """
    YOLO가 예측한 binary mask를 COCO result JSON에서 사용하는 RLE 형식으로 변환한다.

    COCOeval에서 segmentation 평가를 하려면 prediction의 segmentation 필드가
    COCO가 인식할 수 있는 polygon 또는 RLE 형식이어야 한다.

    여기서는 YOLO mask를 binary mask로 받은 뒤,
    pycocotools.mask.encode를 사용하여 RLE로 변환한다.

    입력:
    - binary_mask: 0과 1로 이루어진 numpy mask

    반환:
    - COCO JSON에 저장 가능한 RLE dictionary
      예: {"size": [height, width], "counts": "..."}
    """

    # pycocotools의 RLE encoding은 Fortran-order array를 기대한다.
    binary_mask = np.asfortranarray(binary_mask.astype(np.uint8))

    # binary mask를 RLE로 encoding한다.
    rle = mask_utils.encode(binary_mask)

    # json.dump로 저장하려면 bytes 타입인 counts를 문자열로 바꿔야 한다.
    rle["counts"] = rle["counts"].decode("utf-8")

    return rle


# ============================================================
# 3. YOLO prediction → COCO result JSON 변환
# ============================================================
# YOLO 모델의 예측 결과는 Ultralytics 내부 형식으로 나온다.
# 하지만 기존 평가 함수 evaluate_split은 COCO result JSON을 입력으로 받는다.
#
# 따라서 YOLO 예측 결과를 아래 형식으로 변환해야 한다.
#
# [
#   {
#     "image_id": int,
#     "category_id": int,
#     "bbox": [x, y, width, height],
#     "score": float,
#     "segmentation": RLE
#   },
#   ...
# ]


def run_yolo_inference_to_coco_json(model, image_dir, annotation_path, output_path, title):
    """
    YOLO 모델로 valid 또는 test 이미지에 대해 instance segmentation 추론을 수행하고,
    그 결과를 COCOeval용 prediction JSON으로 저장한다.

    처리 과정:
    1) annotation JSON에서 image 목록과 category 목록을 읽는다.
    2) 각 이미지에 대해 YOLO segmentation predict를 수행한다.
    3) YOLO가 반환한 mask, box, class_id, confidence를 추출한다.
    4) mask를 원본 이미지 크기에 맞춘다.
    5) binary mask를 COCO RLE 형식으로 변환한다.
    6) RLE mask로부터 COCO bbox [x, y, w, h]를 계산한다.
    7) COCO result JSON 형식으로 저장한다.

    입력:
    - model: 학습된 YOLO segmentation 모델
    - image_dir: valid 또는 test 이미지 폴더
    - annotation_path: valid/test COCO annotation JSON 경로
    - output_path: 저장할 COCO prediction JSON 경로
    - title: 출력 로그에 사용할 이름
    """

    print()
    print(f"-------------------------- {title} YOLO INFERENCE --------------------------")

    # COCO annotation JSON 로드
    with open(annotation_path, "r", encoding="utf-8") as f:
        coco = json.load(f)

    images = coco["images"]
    categories = coco["categories"]

    # annotation에 정의된 category id만 유효한 예측으로 인정한다.
    coco_category_ids = set(int(cat["id"]) for cat in categories)

    predictions = []

    # annotation의 image 목록 기준으로 추론한다.
    # 이렇게 해야 image_id가 COCO annotation과 정확히 맞는다.
    for idx, image_info in enumerate(images):
        image_id = int(image_info["id"])
        file_name = image_info["file_name"]

        image_path = image_dir / file_name

        # annotation에는 있지만 실제 이미지 파일이 없으면 건너뛴다.
        if not image_path.exists():
            print("[WARNING] image not found:", image_path)
            continue

        # 원본 이미지 크기를 얻는다.
        # YOLO mask 크기가 원본과 다를 경우 다시 원본 크기로 resize하기 위해 필요하다.
        pil_image = Image.open(image_path).convert("RGB")
        original_width, original_height = pil_image.size

        # YOLO segmentation 추론 실행
        results = model.predict(
            source=str(image_path),
            task="segment",
            conf=CONF_THRES,
            iou=IOU_THRES,
            imgsz=IMGSZ,
            device=DEVICE,
            retina_masks=True,
            verbose=False,
        )

        # 결과가 없으면 다음 이미지로 넘어간다.
        if len(results) == 0:
            continue

        result = results[0]

        # segmentation mask 또는 box 정보가 없으면 사용할 수 없다.
        if result.masks is None or result.boxes is None:
            continue

        # YOLO 결과를 numpy array로 변환한다.
        masks = result.masks.data.cpu().numpy()
        boxes = result.boxes.xyxy.cpu().numpy()
        class_ids = result.boxes.cls.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()

        # YOLO 예측 객체 하나씩 COCO result 형식으로 변환한다.
        for mask, box, class_id, score in zip(masks, boxes, class_ids, confidences):
            # YOLO class id를 COCO category id로 사용한다.
            # 학습 데이터 변환 시 COCO category id를 그대로 YOLO class id로 사용했기 때문에
            # 별도 class mapping이 필요 없다.
            category_id = int(class_id)

            # annotation에 없는 category id는 평가 대상이 아니므로 제외한다.
            if category_id not in coco_category_ids:
                continue

            # YOLO mask를 binary mask로 변환한다.
            binary_mask = mask.astype(np.uint8)

            # mask 크기가 원본 이미지 크기와 다르면 원본 크기로 resize한다.
            # COCOeval은 원본 이미지 기준 mask를 기대하므로 크기를 맞춰야 한다.
            if binary_mask.shape[0] != original_height or binary_mask.shape[1] != original_width:
                binary_mask = cv2.resize(
                    binary_mask,
                    (original_width, original_height),
                    interpolation=cv2.INTER_NEAREST,
                )

            # 빈 mask는 평가에 의미가 없으므로 제외한다.
            if binary_mask.max() == 0:
                continue

            # binary mask를 COCO RLE segmentation으로 변환한다.
            rle = encode_binary_mask(binary_mask)

            # COCO bbox는 [x, y, width, height] 형식이다.
            # mask_utils.toBbox를 사용하면 RLE mask에서 bbox를 계산할 수 있다.
            x, y, w, h = mask_utils.toBbox(rle).tolist()

            # COCO result JSON 형식으로 prediction 하나를 추가한다.
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

        # 진행 상황 출력
        if (idx + 1) % 20 == 0:
            print(f"{idx + 1}/{len(images)} images processed")

    # 전체 prediction list를 COCO result JSON으로 저장한다.
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(predictions, f)

    print()
    print(f"{title} prediction saved:", output_path)
    print(f"{title} num predictions:", len(predictions))


# ============================================================
# 4. 평가 결과 저장 및 baseline 비교
# ============================================================

def save_summary(valid_map50, test_map50):
    """
    기존 pretrained baseline 모델과 YOLO fine-tuning 모델의 성능을 비교하여
    summary CSV 파일로 저장하고, test mAP@50 기준 성공 여부를 출력한다.

    비교 기준:
    - baseline_IS_pretrained_bottom: 1-1에서 평가한 기존 pretrained RF-DETR 모델
    - yolo_seg_assignment02: 1-2에서 fine-tuning한 YOLO11s-seg 모델

    과제 목표:
    - 제공된 pretrained 모델보다 높은 성능 달성
    - 따라서 test mAP@50이 baseline보다 높으면 SUCCESS로 판단한다.
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

    # Excel에서 한글/인코딩 문제가 덜 나도록 utf-8-sig로 저장한다.
    with open(SUMMARY_PATH, "w", encoding="utf-8-sig") as f:
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
    print("summary saved:", SUMMARY_PATH)

    print()

    # 과제의 최종 목표는 test set에서 baseline보다 높은 mAP@50을 얻는 것이다.
    if test_map50 > BASELINE_TEST_MAP50:
        print("RESULT: SUCCESS - YOLO model is better than baseline on test mAP@50.")
    else:
        print("RESULT: NOT YET - YOLO model did not exceed baseline test mAP@50.")


# ============================================================
# 5. main 실행부
# ============================================================

def main():
    """
    전체 평가 흐름을 관리하는 main 함수이다.

    실행 순서:
    1) 학습된 YOLO best.pt 모델이 존재하는지 확인한다.
    2) best.pt를 YOLO 객체로 로드한다.
    3) valid set에 대해 YOLO 추론을 수행하고 COCO prediction JSON을 저장한다.
    4) test set에 대해 YOLO 추론을 수행하고 COCO prediction JSON을 저장한다.
    5) 기존 evaluate_split 함수를 사용해 valid mAP@50과 class metric을 계산한다.
    6) 기존 evaluate_split 함수를 사용해 test mAP@50과 class metric을 계산한다.
    7) baseline 모델과 YOLO 모델의 valid/test mAP@50을 비교하여 summary를 저장한다.
    """

    print()
    print("-------------------------- EVALUATE TRAINED YOLO MODEL --------------------------")
    print("model:", TRAINED_MODEL_PATH)
    print("exists:", TRAINED_MODEL_PATH.exists())
    print("device:", DEVICE)

    # best.pt가 없으면 아직 학습이 완료되지 않은 상태이다.
    if not TRAINED_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"trained model not found: {TRAINED_MODEL_PATH}\n"
            f"먼저 train_instance_segmentation_assignment02.py를 실행하세요."
        )

    # 학습된 YOLO segmentation model 로드
    model = YOLO(str(TRAINED_MODEL_PATH))

    # valid set 추론 결과를 COCO result JSON으로 저장
    run_yolo_inference_to_coco_json(
        model=model,
        image_dir=VALID_DIR,
        annotation_path=VALID_ANN_PATH,
        output_path=VALID_YOLO_PRED_PATH,
        title="VALID",
    )

    # test set 추론 결과를 COCO result JSON으로 저장
    run_yolo_inference_to_coco_json(
        model=model,
        image_dir=TEST_DIR,
        annotation_path=TEST_ANN_PATH,
        output_path=TEST_YOLO_PRED_PATH,
        title="TEST",
    )

    # valid set 평가
    # evaluate_split은 1-1 코드에서 작성한 공통 평가 함수이다.
    # 내부적으로 COCOeval mAP@50과 class별 AP/Precision/Recall을 계산한다.
    valid_map50 = evaluate_split(
        annotation_path=VALID_ANN_PATH,
        prediction_path=VALID_YOLO_PRED_PATH,
        class_metric_path=VALID_YOLO_CLASS_METRIC_PATH,
        title="VALID YOLO ASSIGNMENT02",
    )

    # test set 평가
    test_map50 = evaluate_split(
        annotation_path=TEST_ANN_PATH,
        prediction_path=TEST_YOLO_PRED_PATH,
        class_metric_path=TEST_YOLO_CLASS_METRIC_PATH,
        title="TEST YOLO ASSIGNMENT02",
    )

    # baseline과 YOLO fine-tuning 모델 성능 비교 결과 저장
    save_summary(
        valid_map50=valid_map50,
        test_map50=test_map50,
    )


if __name__ == "__main__":
    main()