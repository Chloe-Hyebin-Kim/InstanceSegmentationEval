# 1-1. 사전학습 Instance Segmentation 모델 Inference 및 평가 보고서

## 1. 과제 목표

본 과제의 목표는 제공된 의상 파트 분류 데이터셋과 사전학습된 instance segmentation 모델 파일 `IS_pretrained_bottom.pt`를 사용하여 validation set과 test set에 대해 inference를 수행하고, 모델의 segmentation 성능을 정량적으로 평가하는 것이다.

평가 항목은 다음과 같다.

1. validation set과 test set에 대한 전체 segmentation mAP@50
2. 클래스별 AP@50
3. 클래스별 GT 개수, 예측 개수
4. 클래스별 TP, FP, FN
5. 클래스별 Precision, Recall

평가는 COCO format annotation과 COCO result JSON format prediction을 기준으로 수행하였다.

---

## 2. 프로젝트 구성

본 구현에서는 실행 진입점과 평가 로직을 분리하였다.

```text
InstanceSegmentationEval/
│
├─ InstanceSegmentationEval.py
├─ pretrained_inference_eval_assignment01.py
├─ IS_pretrained_bottom.pt
│
├─ dataset-coco-seg/
│  ├─ train/
│  ├─ valid/
│  │  └─ _annotations.coco.json
│  └─ test/
│     └─ _annotations.coco.json
│
└─ results/
   ├─ valid_predictions.json
   ├─ test_predictions.json
   ├─ valid_class_metrics.csv
   ├─ test_class_metrics.csv
   └─ summary_map50.csv
```

각 파일의 역할은 다음과 같다.

| 파일명                                         | 역할                                                                     |
| ------------------------------------------- | ---------------------------------------------------------------------- |
| `InstanceSegmentationEval.py`               | 전체 실행 진입점이며 `main()` 함수만 담당                                            |
| `pretrained_inference_eval_assignment01.py` | path 설정, inference 함수, COCO mAP 평가 함수, 클래스별 metric 계산 함수, CSV 저장 함수 포함 |
| `valid_predictions.json`                    | validation set에 대한 COCO result format 예측 결과                            |
| `test_predictions.json`                     | test set에 대한 COCO result format 예측 결과                                  |
| `valid_class_metrics.csv`                   | validation set 클래스별 평가 결과                                              |
| `test_class_metrics.csv`                    | test set 클래스별 평가 결과                                                    |
| `summary_map50.csv`                         | validation/test 전체 mAP@50 요약 결과                                        |

---

## 3. 코드 구조

### 3.1 `InstanceSegmentationEval.py`

`InstanceSegmentationEval.py`는 전체 실행 순서를 담당한다.
이 파일에서는 모델을 로드하고, validation/test set에 대해 inference를 수행한 뒤, mAP@50과 클래스별 metric을 계산한다.

실행 흐름은 다음과 같다.

```text
1. 경로 확인
2. validation/test annotation category 출력
3. 사전학습 모델 로드
4. validation set inference
5. test set inference
6. validation set mAP@50 및 클래스별 metric 평가
7. test set mAP@50 및 클래스별 metric 평가
8. summary_map50.csv 저장
```

### 3.2 `pretrained_inference_eval_assignment01.py`

`pretrained_inference_eval_assignment01.py`는 실제 inference와 evaluation에 필요한 함수들을 포함한다.

주요 함수는 다음과 같다.

| 함수명                                 | 역할                                          |
| ----------------------------------- | ------------------------------------------- |
| `check_paths()`                     | 모델 파일, dataset 경로, annotation 파일 존재 여부 확인   |
| `print_dataset_categories()`        | COCO annotation의 category id와 class name 출력 |
| `encode_binary_mask()`              | binary mask를 COCO RLE format으로 변환           |
| `run_inference()`                   | RF-DETR segmentation 모델로 image inference 수행 |
| `evaluate_map50()`                  | COCOeval을 사용하여 전체 segmentation mAP@50 계산    |
| `evaluate_class_ap50()`             | category별 AP@50 계산                          |
| `evaluate_class_precision_recall()` | 클래스별 TP, FP, FN, Precision, Recall 계산       |
| `save_class_metrics_csv()`          | 클래스별 평가 결과를 CSV 파일로 저장                      |
| `evaluate_split()`                  | 하나의 split에 대해 전체 평가 수행                      |

---

## 4. 데이터셋 및 클래스 구성

데이터셋은 COCO format으로 제공되었으며, validation/test set의 annotation 파일은 각각 다음 위치에 존재한다.

```text
dataset-coco-seg/valid/_annotations.coco.json
dataset-coco-seg/test/_annotations.coco.json
```

확인된 class category는 다음과 같다.

| category_id | class_name  |
| ----------: | ----------- |
|           0 | bottom-0LAX |
|           1 | background  |
|           2 | band        |
|           3 | beltloop    |
|           4 | button      |
|           5 | cuffs       |
|           6 | decoration  |
|           7 | hem         |
|           8 | hook        |
|           9 | lace        |
|          10 | logo        |
|          11 | loop        |
|          12 | placket     |
|          13 | pocket      |
|          14 | quarter     |
|          15 | tag         |
|          16 | zipper      |

단, 실제 GT annotation에서는 `bottom-0LAX`, `background`, `button`, `loop`, `tag` 등 일부 클래스의 GT 개수가 0인 경우가 존재하였다. 이러한 클래스는 AP 계산 시 `N/A`로 처리하였다.

---

## 5. Inference 방법

사전학습 모델은 다음 파일을 사용하였다.

```text
IS_pretrained_bottom.pt
```

모델은 RF-DETR segmentation 모델 구조를 이용하여 로드하였다.

```python
model = RFDETRSegMedium.from_checkpoint(str(MODEL_PATH))
```

각 이미지에 대해 `model.predict()`를 사용하여 instance segmentation 결과를 얻었다.
예측 결과에서 다음 정보를 추출하였다.

1. bounding box
2. class id
3. confidence score
4. segmentation mask

segmentation mask는 COCO evaluation에서 사용할 수 있도록 RLE format으로 변환하였다.

```python
rle = mask_utils.encode(binary_mask)
```

예측 결과는 COCO result JSON format에 맞추어 저장하였다.

```json
{
  "image_id": image_id,
  "category_id": category_id,
  "bbox": [x, y, width, height],
  "score": score,
  "segmentation": rle
}
```

---

## 6. Confidence Threshold 설정 근거

Inference confidence threshold는 `0.1`로 설정하였다.

이 값은 validation set 기준 threshold sweep 결과를 바탕으로 선택하였다.
후보 threshold `0.1`, `0.2`, `0.3`, `0.4`, `0.5`에 대해 validation mAP@50을 비교했을 때, `0.1`에서 가장 높은 validation mAP@50을 보였다.

| confidence threshold | validation mAP@50 | test mAP@50 |
| -------------------: | ----------------: | ----------: |
|                  0.1 |          0.532781 |    0.567696 |
|                  0.2 |          0.498468 |    0.544122 |
|                  0.3 |          0.468227 |    0.497918 |
|                  0.4 |          0.370867 |    0.497918 |
|                  0.5 |          0.356841 |    0.497918 |

따라서 최종 inference threshold는 validation set에서 가장 높은 mAP@50을 보인 `0.1`로 결정하였다.
test set은 threshold 선택에 사용하지 않고, validation set에서 선택된 threshold를 그대로 적용하여 최종 성능 평가에만 사용하였다.

---

## 7. 평가 방법

평가는 `pycocotools`의 `COCOeval`을 사용하였다.
annotation type은 `segm`으로 설정하여 segmentation mask 기준 평가를 수행하였다.

```python
coco_eval = COCOeval(coco_gt, coco_dt, "segm")
```

IoU threshold는 `0.50`으로 고정하였다.

```python
coco_eval.params.iouThrs = np.array([0.5])
```

따라서 본 평가에서 사용한 주요 설정은 다음과 같다.

| 항목                   | 값            |
| -------------------- | ------------ |
| evaluation type      | segmentation |
| annotation type      | `segm`       |
| IoU threshold        | 0.50         |
| confidence threshold | 0.10         |
| metric               | mAP@50       |

---

## 8. 전체 mAP@50 평가 결과

최종 평가 결과는 다음과 같다.

| split      |   mAP@50 |
| ---------- | -------: |
| validation | 0.532781 |
| test       | 0.567696 |

validation set의 segmentation mAP@50은 `0.532781`로 측정되었고, test set의 segmentation mAP@50은 `0.567696`으로 측정되었다.

---

## 9. Validation Set 클래스별 평가 결과

validation set의 클래스별 주요 결과는 다음과 같다.

| class      | GT | PRED |    AP@50 | TP | FP | FN | Precision |   Recall |
| ---------- | -: | ---: | -------: | -: | -: | -: | --------: | -------: |
| band       | 22 |   24 | 0.739824 | 17 |  7 |  5 |  0.708333 | 0.772727 |
| beltloop   |  4 |    8 | 0.504950 |  2 |  6 |  2 |  0.250000 | 0.500000 |
| cuffs      | 12 |   12 | 1.000000 | 12 |  0 |  0 |  1.000000 | 1.000000 |
| decoration |  2 |   11 | 0.000000 |  0 | 11 |  2 |  0.000000 | 0.000000 |
| hem        | 25 |   48 | 0.905697 | 23 | 25 |  2 |  0.479167 | 0.920000 |
| hook       |  1 |    6 | 1.000000 |  1 |  5 |  0 |  0.166667 | 1.000000 |
| lace       |  1 |    6 | 0.000000 |  0 |  6 |  1 |  0.000000 | 0.000000 |
| logo       | 13 |   33 | 0.904348 | 13 | 20 |  0 |  0.393939 | 1.000000 |
| placket    |  1 |    0 | 0.000000 |  0 |  0 |  1 |       N/A | 0.000000 |
| pocket     | 18 |   74 | 0.476141 | 10 | 64 |  8 |  0.135135 | 0.555556 |
| quarter    | 26 |   89 | 0.862410 | 23 | 66 |  3 |  0.258427 | 0.884615 |
| zipper     |  3 |   10 | 0.000000 |  0 | 10 |  3 |  0.000000 | 0.000000 |

validation set에서는 `cuffs`, `hem`, `logo`, `quarter`, `band` 클래스의 AP@50이 상대적으로 높게 나타났다. 반면 `decoration`, `lace`, `placket`, `zipper` 클래스는 AP@50이 0으로 측정되어 성능 개선이 필요한 클래스로 확인되었다.

---

## 10. Test Set 클래스별 평가 결과

test set의 클래스별 주요 결과는 다음과 같다.

| class      | GT | PRED |    AP@50 | TP | FP | FN | Precision |   Recall |
| ---------- | -: | ---: | -------: | -: | -: | -: | --------: | -------: |
| band       |  6 |    6 | 0.663366 |  4 |  2 |  2 |  0.666667 | 0.666667 |
| decoration |  4 |    7 | 0.422442 |  2 |  5 |  2 |  0.285714 | 0.500000 |
| hem        | 10 |   12 | 0.779978 |  8 |  4 |  2 |  0.666667 | 0.800000 |
| logo       |  3 |   11 | 0.834158 |  3 |  8 |  0 |  0.272727 | 1.000000 |
| pocket     |  5 |   23 | 0.273927 |  2 | 21 |  3 |  0.086957 | 0.400000 |
| quarter    |  5 |    9 | 1.000000 |  5 |  4 |  0 |  0.555556 | 1.000000 |
| zipper     |  1 |    2 | 0.000000 |  0 |  2 |  1 |  0.000000 | 0.000000 |

test set에서는 `quarter`, `logo`, `hem`, `band` 클래스가 상대적으로 높은 AP@50을 보였다. 반면 `pocket` 클래스는 prediction 개수에 비해 FP가 많아 precision이 낮게 나타났고, `zipper` 클래스는 TP가 발생하지 않아 AP@50이 0으로 측정되었다.

---

## 11. 결과 분석

전체적으로 validation mAP@50은 `0.532781`, test mAP@50은 `0.567696`으로 측정되었다. test set의 mAP@50이 validation set보다 약간 높게 나타났으며, 이는 test set 내에서 상대적으로 예측이 쉬운 클래스 또는 큰 객체 비중이 존재했을 가능성이 있다.

클래스별 결과를 보면 `cuffs`, `hem`, `logo`, `quarter`, `band`와 같이 형태가 비교적 명확하거나 학습 데이터에서 충분히 등장한 클래스는 높은 AP@50을 보였다. 반면 `decoration`, `lace`, `placket`, `zipper`처럼 GT 개수가 적거나 형태가 작고 다양할 수 있는 클래스에서는 AP@50이 낮게 나타났다.

또한 `pocket`, `quarter`, `logo`, `hem` 클래스에서는 recall은 높지만 precision이 낮은 경우가 확인되었다. 이는 confidence threshold가 낮아 많은 객체 후보를 검출하면서 TP는 확보했지만, 동시에 FP도 증가했기 때문으로 해석할 수 있다. 특히 `pocket` 클래스는 validation set에서 18개의 GT에 대해 74개의 prediction이 발생했고, test set에서도 5개의 GT에 대해 23개의 prediction이 발생하여 FP가 많은 클래스임을 확인할 수 있었다.

---

## 12. 산출 파일

실행 결과 다음 파일들이 생성되었다.

| 파일명                       | 설명                           |
| ------------------------- | ---------------------------- |
| `valid_predictions.json`  | validation set 예측 결과         |
| `test_predictions.json`   | test set 예측 결과               |
| `valid_class_metrics.csv` | validation set 클래스별 평가 결과    |
| `test_class_metrics.csv`  | test set 클래스별 평가 결과          |
| `summary_map50.csv`       | validation/test mAP@50 요약 결과 |

---

## 13. 결론

본 구현에서는 제공된 사전학습 instance segmentation 모델 `IS_pretrained_bottom.pt`를 사용하여 validation set과 test set에 대한 inference 및 평가를 수행하였다. 예측 결과는 COCO result JSON format으로 저장하였고, `pycocotools.COCOeval`을 사용하여 segmentation mAP@50을 계산하였다.

최종적으로 validation set에서는 `mAP@50 = 0.532781`, test set에서는 `mAP@50 = 0.567696`을 얻었다. 클래스별 분석 결과, `cuffs`, `hem`, `logo`, `quarter`, `band` 클래스에서는 비교적 높은 성능을 보였으나, `decoration`, `lace`, `placket`, `zipper`, `pocket` 클래스에서는 성능 개선이 필요한 것으로 확인되었다.

향후 성능 개선을 위해서는 클래스별 confidence threshold 조정, FP가 많은 클래스에 대한 후처리, 부족한 클래스의 데이터 보강, 오픈소스 segmentation 모델 fine-tuning 등의 전략을 적용할 수 있다.
