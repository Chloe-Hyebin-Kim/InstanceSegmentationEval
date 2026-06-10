
from pathlib import Path

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


PROJECT_ROOT = Path.cwd()

TEST_ANN_PATH = PROJECT_ROOT / "dataset-coco-seg" / "dataset-coco-seg" / "test" / "_annotations.coco.json"
TEST_OUTPUT_JSON_PATH = PROJECT_ROOT / "test_predictions.json"

print("-------------------------- TEST mAP@50 --------------------------")
print("TEST_ANN exists:", TEST_ANN_PATH.exists())
print("PRED_JSON exists:", TEST_OUTPUT_JSON_PATH.exists())

gt = COCO(str(TEST_ANN_PATH))
pred = gt.loadRes(str(TEST_OUTPUT_JSON_PATH))

evaluator = COCOeval(gt, pred, iouType="segm")
evaluator.params.iouThrs = np.array([0.5])
evaluator.params.maxDets = [1, 10, 500]

evaluator.evaluate()
evaluator.accumulate()
evaluator.summarize()


#instance segmentation 기준 mAP@50 = 0.616
#(Test set 5장) mAP@50 = 0.592.

#test_predictions.json