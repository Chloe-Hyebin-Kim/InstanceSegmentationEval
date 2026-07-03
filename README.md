# Instance Segmentation 

### 
gpu name: NVIDIA GeForce RTX 5080
* Windows11
* VisualStudio2019
* NVIDIA GeForce RTX 5080
* CUDA 12.8 PyTorch
* Python : 3.11.0
* torch :  ~2.12.0+cpu~ -> torch: 2.11.0+cu128 변경 (CUDA 12.8 wheel)


```python
#설치 패키지

#pip install torch torchvision  # PyTorch #CPU 전용 PyTorch-> CUDA 지원 버전으로변경...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128  #CUDA 지원 버전(CUDA 12.8 wheel)
pip install rfdetr             # 모델 실행용
pip install pycocotools        # COCO mAP 계산용
pip install supervision        # 예측 결과 다루기 편하게 해주는 도구
pip install opencv-python      # 이미지 처리 , COCO mask를 YOLO segmentation polygon 형태로 변환
pip install pillow             # 이미지 열기
pip install matplotlib         # 결과 시각화
pip install tqdm               # 진행률 표시
pip install ultralytics        # YOLO segmentation 학습

```



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

 ---

<br>
<br>



## 1. Istance Segmentation Inference + mAP@50 + 클래스별 예측 정확도 평가

제공된 의상 파트 분류 데이터셋과, 해당 데이터셋에서 사전학습된 instance segmentation을 사용하여 
Instance segmentation 사전학습 모델 파일 (IS_pretraine_bottom.pt)에 대한 inference 코드를 작성하고, 
첨부된 validation, test set에 대해 mAP@50과 클래스 별 예측 정확도를 평가하세요. 

Confidence threshold는 validation set을 이용해 결정하였다. 
0.1, 0.2, 0.3, 0.4, 0.5의 후보 threshold에 대해 validation mAP@50을 비교한 결과, threshold=0.1에서 validation mAP@50이 0.532781로 가장 높았다. 낮은 threshold가 더 좋게 나온 원인에 대해서 분석해 볼때, 모델이 0.2~0.5로 threshold를 올리면 false positive는 줄어들 수 있지만, 그보다 true positive까지 많이 사라져서 recall이 감소한 것으로 추측한다.
COCO AP는 단순 precision만 보는 지표가 아니라, confidence score 순서에 따른 precision-recall curve를 기반으로 계산되는데, threshold를 너무 높이면 맞을 수 있었던 예측도 버려져 AP가 낮아진 것이라고 추측된다.
현재 valid 결과에서도 0.1일 때 recall이 꽤 중요하게 작동하는 것을 볼 수 있다.
예를 들어 validation 기준에서 다음 표를 보면 0.1에서는 FP가 많긴 하지만, 주요 클래스에서 TP를 많이 유지하고 있다. threshold를 올리면 hem, logo, quarter, pocket 같은 클래스의 TP가 줄어들 수 있고, 그 결과 mAP@50이 감소한 것으로 해석할 수 있다.
| class   | GT | PRED | TP | FP | FN |   Recall |
| ------- | -: | ---: | -: | -: | -: | -------: |
| band    | 22 |   24 | 17 |  7 |  5 | 0.772727 |
| cuffs   | 12 |   12 | 12 |  0 |  0 | 1.000000 |
| hem     | 25 |   48 | 23 | 25 |  2 | 0.920000 |
| logo    | 13 |   33 | 13 | 20 |  0 | 1.000000 |
| quarter | 26 |   89 | 23 | 66 |  3 | 0.884615 |

따라서 최종 inference threshold는 0.1로 설정하였고, 동일한 threshold를 test set 평가에 적용하였다. 
그 결과 test set의 mAP@50은 0.567696으로 측정되었다. 



 1) 평가 항목 : COCO format annotation과 COCO result JSON format prediction을 기준으로 수행
       - validation set과 test set에 대한 전체 segmentation mAP@50
       - 클래스별 AP@50
       - 클래스별 GT 개수, 예측 개수
       - 클래스별 TP, FP, FN
       - 클래스별 Precision, Recall
 2) 프로젝트 구성
 본 구현에서는 실행 진입점과 평가 로직을 분리하였다.
    ``` text
    InstanceSegmentationEval/
    │
    ├─ pretrained_inference_eval_assignment01.py   # 함수, 설정, 평가 로직
    ├─ InstanceSegmentationEval.py                 # main 실행 파일
    │
    ├─ IS_pretrained_bottom.pt
    ├─ dataset-coco-seg/
    │  ├─ train/
    │  ├─ valid/
    │  │  └─ _annotations.coco.json
    │  └─ test/
    │     └─ _annotations.coco.json
    │
    └─ results/
       ├─ valid_predictions.json    #validation set 예측 결과
       ├─ test_predictions.json     #test set 예측 결과
       ├─ valid_class_metrics.csv   #validation 클래스별 AP@50, TP, FP, FN, Precision, Recall
       ├─ test_class_metrics.csv    #test 클래스별 AP@50, TP, FP, FN, Precision, Recall
       └─ summary_map50.csv         #valid/test 전체 mAP@50 요약
    ```
    
 4) 결과
    
    Confidence threshold는 validation set을 이용해 결정하였다.<br>
    0.1, 0.2, 0.3, 0.4, 0.5의 후보 threshold에 대해 validation mAP@50을 비교한 결과, <br>
    threshold=0.1에서 Validation set의 segmentation mAP@50은 0.532781로 가장 높았다. <br>
    따라서 최종 inference threshold는 0.1로 설정하였고, 동일한 threshold를 test set 평가에 적용하였다. <br>
    그 결과 test set의 segmentation mAP@50은 0.567696으로 측정되었다.
    | split      |   mAP@50 |
    | ---------- | -------: |
    | validation | 0.532781 |
    | test       | 0.567696 |


 <br>
 <br>
 <br>
 
---

 <br>
 <br>
 <br>
 
## 22. Instance Segmentation 모델 추론 성능 개선

 <br>

기존 제공된 pretrained RF-DETR segmentation 모델을 validation/test set에 대해 평가한 결과, <br>
test set 기준 segmentation mAP@50은 0.567696이었다. 
전체 성능은 어느 정도 확보되어 있었지만, class별 결과를 보면 몇 가지 문제가 확인되었다.

첫째, 일부 class에서 false positive가 많이 발생했다.특히 pocket, quarter, hem, logo와 같은 class는 예측 개수가 GT 개수보다 많아 precision이 낮게 나타났다.  <br>
둘째, zipper, lace, decoration처럼 데이터 수가 적거나 형태가 작은 class에서는 AP가 0에 가깝게 나타났다.  <br>
셋째, 기존 모델은 제공된 pretrained weight를 그대로 사용한 것이기 때문에, 현재 데이터셋의 class 분포와 객체 형태에 완전히 최적화되어 있지는 않다고 판단했다.

1) 개선 방안
따라서 단순히 기존 모델을 그대로 사용하는 것보다, train set을 이용해 현재 데이터셋에 맞게 instance segmentation 모델을 fine-tuning하는 것이 성능 개선에 더 적합하다고 판단했다. 이를 위한 성능 개선 방법으로는 새로운 instance segmentation 모델을 학습하는 방식을 선택했다. 후보로는 RF-DETR 추가 학습, Mask R-CNN 계열 모델, YOLO segmentation 모델이 있었다. 그중 YOLO11s-seg 모델을 선택하였다.

2)
# Instance Segmentation 

### 
gpu name: NVIDIA GeForce RTX 5080
* Windows11
* VisualStudio2019
* NVIDIA GeForce RTX 5080
* CUDA 12.8 PyTorch
* Python : 3.11.0
* torch :  ~2.12.0+cpu~ -> torch: 2.11.0+cu128 변경 (CUDA 12.8 wheel)


```python
#설치 패키지

#pip install torch torchvision  # PyTorch #CPU 전용 PyTorch-> CUDA 지원 버전으로변경...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128  #CUDA 지원 버전(CUDA 12.8 wheel)
pip install rfdetr             # 모델 실행용
pip install pycocotools        # COCO mAP 계산용
pip install supervision        # 예측 결과 다루기 편하게 해주는 도구
pip install opencv-python      # 이미지 처리 , COCO mask를 YOLO segmentation polygon 형태로 변환
pip install pillow             # 이미지 열기
pip install matplotlib         # 결과 시각화
pip install tqdm               # 진행률 표시
pip install ultralytics        # YOLO segmentation 학습

```



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

 ---

<br>
<br>



## 1. Istance Segmentation Inference + mAP@50 + 클래스별 예측 정확도 평가

제공된 의상 파트 분류 데이터셋과, 해당 데이터셋에서 사전학습된 instance segmentation을 사용하여 
Instance segmentation 사전학습 모델 파일 (IS_pretraine_bottom.pt)에 대한 inference 코드를 작성하고, 
첨부된 validation, test set에 대해 mAP@50과 클래스 별 예측 정확도를 평가하세요. 

Confidence threshold는 validation set을 이용해 결정하였다. 
0.1, 0.2, 0.3, 0.4, 0.5의 후보 threshold에 대해 validation mAP@50을 비교한 결과, threshold=0.1에서 validation mAP@50이 0.532781로 가장 높았다. 낮은 threshold가 더 좋게 나온 원인에 대해서 분석해 볼때, 모델이 0.2~0.5로 threshold를 올리면 false positive는 줄어들 수 있지만, 그보다 true positive까지 많이 사라져서 recall이 감소한 것으로 추측한다.
COCO AP는 단순 precision만 보는 지표가 아니라, confidence score 순서에 따른 precision-recall curve를 기반으로 계산되는데, threshold를 너무 높이면 맞을 수 있었던 예측도 버려져 AP가 낮아진 것이라고 추측된다.
현재 valid 결과에서도 0.1일 때 recall이 꽤 중요하게 작동하는 것을 볼 수 있다.
예를 들어 validation 기준에서 다음 표를 보면 0.1에서는 FP가 많긴 하지만, 주요 클래스에서 TP를 많이 유지하고 있다. threshold를 올리면 hem, logo, quarter, pocket 같은 클래스의 TP가 줄어들 수 있고, 그 결과 mAP@50이 감소한 것으로 해석할 수 있다.
| class   | GT | PRED | TP | FP | FN |   Recall |
| ------- | -: | ---: | -: | -: | -: | -------: |
| band    | 22 |   24 | 17 |  7 |  5 | 0.772727 |
| cuffs   | 12 |   12 | 12 |  0 |  0 | 1.000000 |
| hem     | 25 |   48 | 23 | 25 |  2 | 0.920000 |
| logo    | 13 |   33 | 13 | 20 |  0 | 1.000000 |
| quarter | 26 |   89 | 23 | 66 |  3 | 0.884615 |

따라서 최종 inference threshold는 0.1로 설정하였고, 동일한 threshold를 test set 평가에 적용하였다. 
그 결과 test set의 mAP@50은 0.567696으로 측정되었다. 



 1) 평가 항목 : COCO format annotation과 COCO result JSON format prediction을 기준으로 수행
       - validation set과 test set에 대한 전체 segmentation mAP@50
       - 클래스별 AP@50
       - 클래스별 GT 개수, 예측 개수
       - 클래스별 TP, FP, FN
       - 클래스별 Precision, Recall
 2) 프로젝트 구성
 본 구현에서는 실행 진입점과 평가 로직을 분리하였다.
    ``` text
    InstanceSegmentationEval/
    │
    ├─ pretrained_inference_eval_assignment01.py   # 함수, 설정, 평가 로직
    ├─ InstanceSegmentationEval.py                 # main 실행 파일
    │
    ├─ IS_pretrained_bottom.pt
    ├─ dataset-coco-seg/
    │  ├─ train/
    │  ├─ valid/
    │  │  └─ _annotations.coco.json
    │  └─ test/
    │     └─ _annotations.coco.json
    │
    └─ results/
       ├─ valid_predictions.json    #validation set 예측 결과
       ├─ test_predictions.json     #test set 예측 결과
       ├─ valid_class_metrics.csv   #validation 클래스별 AP@50, TP, FP, FN, Precision, Recall
       ├─ test_class_metrics.csv    #test 클래스별 AP@50, TP, FP, FN, Precision, Recall
       └─ summary_map50.csv         #valid/test 전체 mAP@50 요약
    ```
    
 4) 결과
    
    Confidence threshold는 validation set을 이용해 결정하였다.<br>
    0.1, 0.2, 0.3, 0.4, 0.5의 후보 threshold에 대해 validation mAP@50을 비교한 결과, <br>
    threshold=0.1에서 Validation set의 segmentation mAP@50은 0.532781로 가장 높았다. <br>
    따라서 최종 inference threshold는 0.1로 설정하였고, 동일한 threshold를 test set 평가에 적용하였다. <br>
    그 결과 test set의 segmentation mAP@50은 0.567696으로 측정되었다.
    | split      |   mAP@50 |
    | ---------- | -------: |
    | validation | 0.532781 |
    | test       | 0.567696 |


 <br>
 <br>
 <br>
 
---

 <br>
 <br>
 <br>
 
## 2. Instance Segmentation 모델 추론 성능 개선

 <br>

기존 제공된 pretrained RF-DETR segmentation 모델을 validation/test set에 대해 평가한 결과, <br>
test set 기준 segmentation mAP@50은 0.567696이었다. 
전체 성능은 어느 정도 확보되어 있었지만, class별 결과를 보면 몇 가지 문제가 확인되었다.

첫째, 일부 class에서 false positive가 많이 발생했다.특히 pocket, quarter, hem, logo와 같은 class는 예측 개수가 GT 개수보다 많아 precision이 낮게 나타났다.  <br>
둘째, zipper, lace, decoration처럼 데이터 수가 적거나 형태가 작은 class에서는 AP가 0에 가깝게 나타났다.  <br>
셋째, 기존 모델은 제공된 pretrained weight를 그대로 사용한 것이기 때문에, 현재 데이터셋의 class 분포와 객체 형태에 완전히 최적화되어 있지는 않다고 판단했다.

1) 개선 방안
따라서 단순히 기존 모델을 그대로 사용하는 것보다, train set을 이용해 현재 데이터셋에 맞게 instance segmentation 모델을 fine-tuning하는 것이 성능 개선에 더 적합하다고 판단했다. 이를 위한 성능 개선 방법으로는 새로운 instance segmentation 모델을 학습하는 방식을 선택했다. 후보로는 RF-DETR 추가 학습, Mask R-CNN 계열 모델, YOLO segmentation 모델이 있었다. 그중 YOLO11s-seg 모델을 선택하였다.

2) 개선 과정
   - fine tuning
     먼저 COCO segmentation 형식으로 제공된 train, validation, test annotation을 YOLO segmentation 학습 형식으로 변환했다. COCO    annotation에는 polygon 또는 RLE segmentation 정보가 포함되어 있으므로, 이를 YOLO segmentation label format에 맞게 class id와 normalized polygon 좌표로 변환하였다.
     학습 데이터 변환 과정에서 train annotation JSON의 metadata 부분에 문법 오류가 있어 json.decoder.JSONDecodeError가 발생했다. 오류는 info 항목 안에 key 없이 문자열이 들어가 있었기 때문에 발생했다. 이 부분은 실제 annotation 데이터가 아니라 metadata였으므로, "description" key를 추가하여 정상 JSON 형식으로 수정하였다. 이후 train annotation은 정상적으로 로드되었고, train 195장, validation 20장, test 5장의 데이터셋을 사용할 수 있었다.
     그 다음 YOLO11s-seg pretrained weight를 불러와 현재 데이터셋의 17개 class에 맞게 fine-tuning하였다. 학습은 RTX 5080 GPU 환경에서 진행했으며, 총 100 epochs 동안 학습하였다. 학습 결과 best.pt 모델이 저장되었고, 이를 사용하여 validation/test set에 대해 다시 추론을 수행하였다.
     마지막으로 YOLO 자체 평가 결과만 사용하지 않고, 기존 pretrained RF-DETR 모델과 동일한 방식으로 비교하기 위해 COCOeval 기반 segmentation mAP@50 평가를 수행했다. 이를 위해 YOLO 예측 결과를 COCO prediction JSON 형식으로 변환한 뒤, 기존 평가 코드와 같은 조건에서 mAP@50을 계산하였다.

   - 후처리 및 threshold 조정
     처음 YOLO 모델을 평가할 때 confidence threshold를 0.1로 설정했을 때는 test mAP@50이 0.563861로 기존 baseline인 0.567696보다 약간 낮았다. 그러나 COCO AP 평가는 confidence score 순위 전체를 활용하는 방식이므로, 너무 높은 threshold를 사용하면 AP 계산에 필요한 낮은 confidence 후보들이 제거될 수 있다.
     따라서 평가용 confidence threshold를 0.001로 낮추어 다시 평가하였다. 이 설정을 적용하자 validation prediction 수는 142개에서 555개로 증가했고, test prediction 수는 33개에서 103개로 증가했다. 그 결과 validation mAP@50과 test mAP@50이 모두 상승하였다.



3)성능 비교 결과

| 모델                    | VALID mAP@50 | TEST mAP@50 |
| --------------------- | -----------: | ----------: |
| 기존 pretrained RF-DETR |     0.532781 |    0.567696 |
| 학습한 YOLO11s-seg       |     0.603551 |    0.570226 |

Validation set에서는 mAP@50이 0.532781에서 0.603551로 향상되었다. 향상폭은 다음과 같다. <br>
0.603551 - 0.532781 = +0.070770 <br>
Test set에서는 mAP@50이 0.567696에서 0.570226으로 향상되었다. 향상폭은 다음과 같다. <br>
0.570226 - 0.567696 = +0.002530 <br>
Test set에서의 향상폭은 크지는 않지만, 과제 목표였던 “제공된 사전학습 모델보다 높은 성능 달성” 조건은 만족하였다.

