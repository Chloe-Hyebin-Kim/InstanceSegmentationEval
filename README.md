# Instance Segmentation 

### 
gpu name: NVIDIA GeForce RTX 5080
* Windows11
* VisualStudio2019
* NVIDIA GeForce RTX 5080
* CUDA 12.8 PyTorch
* Python : 3.11.0
* torch :  ~2.12.0+cpu~ -> torch: 2.11.0+cu128 변경 (CUDA 12.8 wheel)


```bat
실행 순서

1. 가상환경 활성화
cd /d .....path....
.venv\Scripts\activate

2. 필요한 패키지 설치
pip install ultralytics opencv-python pycocotools

GPU PyTorch가 아닌 경우 CUDA PyTorch 설치:
pip uninstall -y torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

3. 01번 baseline 평가
python 01_pretrained_inference_eval.py

4. 02번 COCO JSON 보정
python 02_fix_coco_json.py

5. 02번 YOLO 학습
python 02_train_yolo_segmentation.py

6. 02번 YOLO 평가
python 02_evaluate_yolo_segmentation.py


```



```text
D:\git\InstanceSegmentationEval\InstanceSegmentationEval
│
├─ .venv
├─ IS_pretrained_bottom.pt
│
├─ 01_pretrained_inference_eval.py
├─ 02_fix_coco_json.py
├─ 02_train_yolo_segmentation.py
├─ 02_evaluate_yolo_segmentation.py
│
├─ config.py
├─ utils.py
├─ inference.py
├─ evaluation.py
├─ metrics.py
│
├─ dataset-coco-seg
│  ├─ train
│  ├─ valid
│  └─ test
│
├─ dataset-yolo-seg-assignment02
│  ├─ images
│  ├─ labels
│  └─ data.yaml
│
├─ runs_assignment02
│  └─ yolo_seg_assignment02
│     └─ weights
│        ├─ best.pt
│        └─ last.pt
│
└─ results
   ├─ valid_predictions.json
   ├─ test_predictions.json
   ├─ valid_class_metrics.csv
   ├─ test_class_metrics.csv
   ├─ summary_map50.csv
   ├─ valid_yolo_assignment02_predictions.json
   ├─ test_yolo_assignment02_predictions.json
   ├─ valid_yolo_assignment02_class_metrics.csv
   ├─ test_yolo_assignment02_class_metrics.csv
   └─ summary_assignment02_yolo.csv
```
| 파일                                 | 역할                                              |
| ---------------------------------- | ----------------------------------------------- |
| `config.py`                        | 전체 경로, threshold, batch, epoch, baseline 성능값 설정 |
| `utils.py`                         | 경로 확인, JSON 보정, mask encoding, COCO → YOLO 변환   |
| `inference.py`                     | RF-DETR 추론, YOLO 추론, COCO prediction JSON 저장    |
| `evaluation.py`                    | COCOeval mAP@50 평가 흐름                           |
| `metrics.py`                       | class별 AP@50, TP/FP/FN, Precision, Recall 계산    |
| `01_pretrained_inference_eval.py`  | 01번: 제공 pretrained RF-DETR 모델 평가                |
| `02_fix_coco_json.py`              | 02번 학습 전 COCO JSON 오류 보정                        |
| `02_train_yolo_segmentation.py`    | 02번: YOLO11s-seg fine-tuning                    |
| `02_evaluate_yolo_segmentation.py` | 02번: YOLO 모델 COCOeval 평가 및 baseline 비교          |

 ---

<br>
<br>




