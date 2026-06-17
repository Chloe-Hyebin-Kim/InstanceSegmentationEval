# Instance Segmentation 

### 

* VisualStudio2019
* torch :  2.12.0+cpu
* Python : 3.11.0

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
       ├─ valid_predictions.json
       ├─ test_predictions.json
       ├─ valid_class_metrics.csv
       ├─ test_class_metrics.csv
       └─ summary_map50.csv
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


