
import json
from pathlib import Path


PROJECT_ROOT = Path(r"D:\git\InstanceSegmentationEval\InstanceSegmentationEval")

ANNOTATION_PATHS = [
    PROJECT_ROOT / "dataset-coco-seg" / "train" / "_annotations.coco.json",
    PROJECT_ROOT / "dataset-coco-seg" / "valid" / "_annotations.coco.json",
    PROJECT_ROOT / "dataset-coco-seg" / "test" / "_annotations.coco.json",
]


def fix_json_file(annotation_path):
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


def main():
    for annotation_path in ANNOTATION_PATHS:
        if not annotation_path.exists():
            raise FileNotFoundError(annotation_path)

        fix_json_file(annotation_path)


if __name__ == "__main__":
    main()