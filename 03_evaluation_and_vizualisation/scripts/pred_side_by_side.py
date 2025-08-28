# scripts/pred_side_by_side.py
import json
from pathlib import Path
from typing import Optional  # <-- for Python 3.8/3.9 compatibility
import random
import cv2
import matplotlib.pyplot as plt

BDD10_NAMES = [
    "person", "rider", "car", "bus", "truck",
    "bike", "motor", "traffic light", "traffic sign", "train"
]

def load_gt(gt_json):
    data = json.loads(Path(gt_json).read_text(encoding="utf-8"))
    id2name = {int(im["id"]): im["file_name"] for im in data["images"]}
    gt_by_img = {}
    for a in data["annotations"]:
        gt_by_img.setdefault(int(a["image_id"]), []).append(a)
    return id2name, gt_by_img

def load_dt(dt_json):
    dts = json.loads(Path(dt_json).read_text(encoding="utf-8"))
    dt_by_img = {}
    for d in dts:
        iid = int(d.get("image_id"))
        dt_by_img.setdefault(iid, []).append(d)
    return dt_by_img

def draw(img, anns, color, is_gt=False):
    for a in anns:
        bbox = a.get("bbox", [])
        if not bbox or len(bbox) < 4:
            continue
        x, y, w, h = bbox
        # two points: (x1,y1) and (x2,y2)
        p1 = (int(x), int(y))
        p2 = (int(x + w), int(y + h))
        cv2.rectangle(img, p1, p2, color, 2)

        cid = int(a.get("category_id", 0))
        name = BDD10_NAMES[cid - 1] if 1 <= cid <= len(BDD10_NAMES) else str(cid)
        score = a.get("score", 1.0)
        label = name if is_gt else f"{name} {float(score):.2f}"
        cv2.putText(img, label, (p1[0], max(15, p1[1] - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

def find_image(images_dir: Path, file_name: str) -> Optional[Path]:
    """Try direct join first; if missing, search recursively."""
    p = images_dir / Path(file_name).name
    if p.exists():
        return p
    hits = list(images_dir.rglob(Path(file_name).name))
    return hits[0] if hits else None

def main(images_dir, gt_json, dt_json, out_dir, n=10, conf_thresh=0.2):
    images_dir = Path(images_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    id2name, gt_by_img = load_gt(gt_json)
    dt_by_img = load_dt(dt_json)

    # only images present in both GT and DT
    img_ids = list(set(gt_by_img.keys()) & set(dt_by_img.keys()))
    if not img_ids:
        print("[WARN] No overlapping images between GT and DT. Check image_id alignment.")
        return

    random.shuffle(img_ids)
    img_ids = img_ids[:n]

    for iid in img_ids:
        fn = Path(id2name[iid]).name  # keep basename for lookup on disk
        path = find_image(images_dir, fn)
        if path is None:
            print(f"[WARN] Could not find image '{fn}' under {images_dir}")
            continue

        img = cv2.imread(str(path))
        if img is None:
            print(f"[WARN] Failed to read image: {path}")
            continue

        gt_anns = gt_by_img.get(iid, [])
        dt_anns = [d for d in dt_by_img.get(iid, []) if float(d.get("score", 1.0)) >= conf_thresh]

        left = img.copy()
        draw(left, gt_anns, (0, 255, 0), is_gt=True)      # green (BGR)

        right = img.copy()
        draw(right, dt_anns, (0, 0, 255), is_gt=False)    # red (BGR)

        concat = cv2.hconcat([left, right])
        out_path = out_dir / fn
        cv2.imwrite(str(out_path), concat)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--images_dir", required=True)
    ap.add_argument("--gt_json", required=True)
    ap.add_argument("--dt_json", required=True)
    ap.add_argument("--out_dir", default="outputs/viz_side_by_side_yolov8n")
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--conf_thresh", type=float, default=0.2)
    args = ap.parse_args()
    main(**vars(args))

