
from utils import fix_all_coco_json_files


# ============================================================
# 02번 학습 전 COCO JSON 오류 보정 스크립트
# ============================================================
# train/_annotations.coco.json의 metadata가 깨져 있으면
# YOLO dataset 변환 단계에서 json.load 오류가 발생한다.
# 이 파일은 train/valid/test annotation을 검사하고 필요한 경우 자동 보정한다.


def main():
    print()
    print("-------------------------- FIX COCO JSON START --------------------------")
    fix_all_coco_json_files()
    print()
    print("-------------------------- FIX COCO JSON DONE --------------------------")


if __name__ == "__main__":
    main()
