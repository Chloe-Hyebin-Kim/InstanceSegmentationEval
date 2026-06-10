
from pathlib import Path

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


PROJECT_ROOT = Path.cwd()

VALID_ANN_PATH = PROJECT_ROOT / "dataset-coco-seg" / "dataset-coco-seg" / "valid" / "_annotations.coco.json"
VALID_OUTPUT_JSON_PATH = PROJECT_ROOT / "valid_predictions.json"

print("-------------------------- VALID mAP@50 --------------------------")
print("VALID_ANN exists:", VALID_ANN_PATH.exists())
print("PRED_JSON exists:", VALID_OUTPUT_JSON_PATH.exists())

gt = COCO(str(VALID_ANN_PATH))
pred = gt.loadRes(str(VALID_OUTPUT_JSON_PATH))

evaluator = COCOeval(gt, pred, iouType="segm")
evaluator.params.iouThrs = np.array([0.5])
evaluator.params.maxDets = [1, 10, 500]

evaluator.evaluate()
evaluator.accumulate()
evaluator.summarize()


#valid 평가 mAP@50 = 0.616
