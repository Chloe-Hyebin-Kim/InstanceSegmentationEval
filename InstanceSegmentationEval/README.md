# 1. 사전학습 Instance Segmentation 모델 Inference 및 평가

<br>

## 1-1. 과제 목표
본 과제의 목표는 제공된 의상 파트 분류 데이터셋과 사전학습된 instance segmentation 모델 파일 `IS_pretrained_bottom.pt`를 사용하여 validation set과 test set에 대해 inference를 수행하고, 모델의 segmentation 성능을 정량적으로 평가하는 것이다.
평가 항목은 다음과 같다.
   1. validation set과 test set에 대한 전체 segmentation mAP@50
   2. 클래스별 AP@50
   3. 클래스별 GT 개수, 예측 개수
   4. 클래스별 TP, FP, FN
   5. 클래스별 Precision, Recall

평가는 COCO format annotation과 COCO result JSON format prediction을 기준으로 수행하였다.

<br>
<br>

## 1-2. 프로젝트 구성

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


<br>
<br>



## 1-3. 코드 구조

### 1-3-1 `InstanceSegmentationEval.py`

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

### 1-3-2 `pretrained_inference_eval_assignment01.py`

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


<br>
<br>



## 1-4. 데이터셋 및 클래스 구성

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


<br>
<br>


## 1-5. Inference 방법

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


<br>
<br>



## 1-6. Confidence Threshold 설정 근거

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


<br>
<br>


## 1-7. 평가 방법

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


<br>
<br>



## 1-8. 전체 mAP@50 평가 결과

최종 평가 결과는 다음과 같다.

| split      |   mAP@50 |
| ---------- | -------: |
| validation | 0.532781 |
| test       | 0.567696 |

validation set의 segmentation mAP@50은 `0.532781`로 측정되었고, test set의 segmentation mAP@50은 `0.567696`으로 측정되었다.


<br>
<br>



## 1-9. Validation Set 클래스별 평가 결과

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


<br>
<br>



## 1-10. Test Set 클래스별 평가 결과

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


<br>
<br>



## 1-11. 결과 분석

전체적으로 validation mAP@50은 `0.532781`, test mAP@50은 `0.567696`으로 측정되었다. test set의 mAP@50이 validation set보다 약간 높게 나타났으며, 이는 test set 내에서 상대적으로 예측이 쉬운 클래스 또는 큰 객체 비중이 존재했을 가능성이 있다.

클래스별 결과를 보면 `cuffs`, `hem`, `logo`, `quarter`, `band`와 같이 형태가 비교적 명확하거나 학습 데이터에서 충분히 등장한 클래스는 높은 AP@50을 보였다. 반면 `decoration`, `lace`, `placket`, `zipper`처럼 GT 개수가 적거나 형태가 작고 다양할 수 있는 클래스에서는 AP@50이 낮게 나타났다.

또한 `pocket`, `quarter`, `logo`, `hem` 클래스에서는 recall은 높지만 precision이 낮은 경우가 확인되었다. 이는 confidence threshold가 낮아 많은 객체 후보를 검출하면서 TP는 확보했지만, 동시에 FP도 증가했기 때문으로 해석할 수 있다. 특히 `pocket` 클래스는 validation set에서 18개의 GT에 대해 74개의 prediction이 발생했고, test set에서도 5개의 GT에 대해 23개의 prediction이 발생하여 FP가 많은 클래스임을 확인할 수 있었다.

<br>
<br>


## 1-12. 산출 파일

실행 결과 다음 파일들이 생성되었다.

| 파일명                       | 설명                           |
| ------------------------- | ---------------------------- |
| `valid_predictions.json`  | validation set 예측 결과         |
| `test_predictions.json`   | test set 예측 결과               |
| `valid_class_metrics.csv` | validation set 클래스별 평가 결과    |
| `test_class_metrics.csv`  | test set 클래스별 평가 결과          |
| `summary_map50.csv`       | validation/test mAP@50 요약 결과 |


<br>
<br>


## 1-13. 결론

본 구현에서는 제공된 사전학습 instance segmentation 모델 `IS_pretrained_bottom.pt`를 사용하여 validation set과 test set에 대한 inference 및 평가를 수행하였다. 예측 결과는 COCO result JSON format으로 저장하였고, `pycocotools.COCOeval`을 사용하여 segmentation mAP@50을 계산하였다.

최종적으로 validation set에서는 `mAP@50 = 0.532781`, test set에서는 `mAP@50 = 0.567696`을 얻었다. 클래스별 분석 결과, `cuffs`, `hem`, `logo`, `quarter`, `band` 클래스에서는 비교적 높은 성능을 보였으나, `decoration`, `lace`, `placket`, `zipper`, `pocket` 클래스에서는 성능 개선이 필요한 것으로 확인되었다.

향후 성능 개선을 위해서는 클래스별 confidence threshold 조정, FP가 많은 클래스에 대한 후처리, 부족한 클래스의 데이터 보강, 오픈소스 segmentation 모델 fine-tuning 등의 전략을 적용할 수 있다.


<br>
<br>
<br>


---

<br>
<br>
<br>



# 2. 추론성능 개선 전략 및 재학습 전략
<br>

## 2-1. 현재 사전학습 모델 성능 요약
평가 결과, 제공된 사전학습 모델 IS_pretrained_bottom.pt의 성능은 다음과 같이 측정되었다. <br>
| split      |   mAP@50 |
| ---------- | -------: |
| validation | 0.532781 |
| test       | 0.567696 |

<br>
<br>


## 2-2. 클래스별 성능 분석
validation set 기준으로 성능이 좋은 클래스와 낮은 클래스를 나누면 다음과 같다.

### 2-2-1 성능이 좋은 클래스
cuffs, hem, logo, quarter, band는 AP@50이 비교적 높게 나타났다. <br>
그러나 hem, logo, quarter는 recall은 높지만 precision이 낮다. <br>
즉, 실제 객체를 많이 찾기는 하지만 false positive도 많이 발생하고 있다.

| class   |    AP@50 | Precision |   Recall |
| ------- | -------: | --------: | -------: |
| cuffs   | 1.000000 |  1.000000 | 1.000000 |
| hook    | 1.000000 |  0.166667 | 1.000000 |
| hem     | 0.905697 |  0.479167 | 0.920000 |
| logo    | 0.904348 |  0.393939 | 1.000000 |
| quarter | 0.862410 |  0.258427 | 0.884615 |
| band    | 0.739824 |  0.708333 | 0.772727 |


### 2-2-2 성능이 낮은 클래스
특히 pocket은 validation set에서 GT 18개에 대해 prediction이 74개 발생했고, TP는 10개, FP는 64개로 precision이 0.135135에 불과하다.<br>
test set에서도 GT 5개에 대해 prediction 23개, TP 2개, FP 21개로 precision이 0.086957이다. <br>
따라서 pocket은 false positive를 줄이는 전략이 필요하다.

| class      |    AP@50 | 문제           |
| ---------- | -------: | ------------ |
| decoration | 0.000000 | TP 없음, FP 많음 |
| lace       | 0.000000 | TP 없음        |
| placket    | 0.000000 | 예측 없음        |
| zipper     | 0.000000 | TP 없음        |
| pocket     | 0.476141 | FP가 매우 많음    |




<br>
<br>

## 2-3. 사전학습 모델 기반 추론성능 개선 전략
재학습 전에 먼저 기존 모델의 inference/post-processing을 개선한다.

### 2-3-1 Class-wise confidence threshold 적용
현재 전체 클래스에 동일하게 confidence threshold = 0.1을 적용하였다. <br>
이 값은 전체 validation mAP@50 기준으로 가장 좋았지만, 클래스별로 보면 false positive가 많은 클래스가 존재한다.<br>

예를 들어 validation set에서 아래 표에 있는 클래스들은 recall은 높지만 FP가 많으므로 threshold를 높이는 실험이 필요하다.
| class   | GT | PRED | TP | FP | Precision |
| ------- | -: | ---: | -: | -: | --------: |
| pocket  | 18 |   74 | 10 | 64 |  0.135135 |
| quarter | 26 |   89 | 23 | 66 |  0.258427 |
| logo    | 13 |   33 | 13 | 20 |  0.393939 |
| hem     | 25 |   48 | 23 | 25 |  0.479167 |

   - 적용 방향
     FP가 많은 클래스 → threshold 증가
     FN이 많은 클래스 → threshold 감소 또는 유지
     AP가 0인 클래스 → threshold보다 데이터/학습 문제 우선 확인

   - 기대 효과
     pocket, quarter, logo, hem 클래스의 FP 감소
     precision 향상
     전체 mAP@50 개선 가능




### 2-3-2 클래스별 후처리 적용
pocket, quarter, logo, hem처럼 prediction이 많은 클래스는 작은 mask noise나 중복 mask가 FP로 이어질 수 있다. <br>
따라서 클래스별로 후처리를 적용한다. 단, 작은 객체 클래스인 hook, lace, zipper에는 강한 area filtering을 적용하면 오히려 TP가 사라질 수 있다.
   - 적용 방향
   1. 작은 mask area 제거
   2. 너무 길거나 비정상적인 bbox ratio 제거
   3. connected component 기반 작은 조각 제거
   4. 클래스별 NMS threshold 조정
   5. 같은 이미지 안에서 과도하게 많은 prediction 제한


### 2-3-3 validation 기반 threshold sweep 자동화
현재 threshold 0.1은 실험 결과 가장 좋았기 때문에 사용하였다.  <br>
하지만 모델을 재학습하면 optimal threshold도 바뀔 수 있다.  <br>
따라서 재학습 모델마다 validation set에서 threshold sweep을 다시 수행한다. <br>
   - 평가 기준
   1. validation mAP@50 최대
   2. 클래스별 AP@50 개선 여부
   3. FP가 많은 클래스의 precision 개선 여부
   4. test set은 최종 1회 평가에만 사용



<br>
<br>


## 2-4. 사전학습 모델 기반 추론성능 개선 전략


### 2-4-1 RF-DETR Segmentation fine-tuning 
동일 계열 모델을 train set으로 fine-tuning하는 것이 가장 자연스러움. 
단, 데이터셋이 작다면 큰 모델은 overfitting될 수 있으므로 validation mAP@50 기준으로 선택.
  - 현재 baseline과 모델 계열이 유사함
  - COCO format dataset을 활용하기 좋음
  - 기존 inference/evaluation 코드 재사용 가능
  - 제공 모델보다 높은 성능을 목표로 fine-tuning하기 적합


### 2-4-2 YOLO Segmentation 계열 fine-tuning
작은 데이터셋에서 빠르게 baseline을 만들기 좋은 Ultralytics YOLO 사용.
단, 데이터셋 변환이 필요. 
   COCO segmentation format→ YOLO segmentation format
   
   - 학습과 추론 코드가 단순함
   - 작은 데이터셋에서 빠르게 baseline을 만들기 좋음
   - inference 속도가 빠른 편임
   - ONNX/TensorRT export가 쉬워 1-3 실배포 속도 개선 전략과 연결하기 좋음

### 2-4-3 Detectron2 Mask R-CNN fine-tuning
two-stage model이라 작은 객체나 복잡한 mask에서 안정적인 경향을 보이는 Detectron2 (Meta/FAIR의 object detection 및 segmentation framework) 을 사용. Mask R-CNN, PointRend, TensorMask 등을 포함하는 baseline.

   다음 논문에서 Faster R-CNN에 mask branch를 추가하여 object detection과 mask prediction을 동시에 수행하는 구조를 제안함.
   (https://arxiv.org/abs/1703.06870?utm_source=chatgpt.com )

   장점
   - COCO format과 잘 맞음
   - 클래스별 분석과 디버깅이 쉬움
   - two-stage model이라 작은 객체나 복잡한 mask에서 안정적일 수 있음
   단점
   - YOLO 계열보다 inference가 느릴 수 있음
   - Windows 환경 세팅이 번거로울 수 있음
   - CUDA/PyTorch/Detectron2 버전 호환 확인 필요
