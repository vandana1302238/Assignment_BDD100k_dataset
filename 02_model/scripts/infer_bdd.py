"""Inference using pretrained model yolov8n for bdd dataset"""
import json
from pathlib import Path
from ultralytics import YOLO
from tqdm import tqdm

BDD10_NAMES = [
    "person", "rider", "car", "bus", "truck",
    "bike", "motor", "traffic light", "traffic sign", "train"
]
NAME2ID = {n: i+1 for i, n in enumerate(BDD10_NAMES)}

CLASS_CONF = {
    "traffic light": 0.12, "traffic sign": 0.12,
    "person": 0.15, "bike": 0.15, "rider": 0.15, "motor": 0.15,
    "car": 0.20, "bus": 0.22, "truck": 0.22, "train": 0.22,
}
GLOBAL_IOU_NMS = 0.6


def build_class_map(model):
    to_bdd = {}
    for midx, mname in model.names.items():
        norm = str(mname).lower().replace("-", " ").replace("_", " ")
        for bname, cid in NAME2ID.items():
            if bname in norm:
                to_bdd[midx] = cid
                break
    return to_bdd


def infer(weights, images_dir, out_json, imgsz=1536, device=None, tta=True):
    images_dir = Path(images_dir)
    out_json = Path(out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)

    model = YOLO(weights)
    to_bdd_cat = build_class_map(model)

    images = sorted(list(images_dir.glob("*.jpg")))
    results = []

    for img_id, p in enumerate(tqdm(images, desc="Infer"), start=1):
        r = model.predict(
            source=str(p),
            imgsz=imgsz,
            conf=0.12,
            iou=GLOBAL_IOU_NMS,
            device=device,
            augment=bool(tta),
            verbose=False
        )[0]
        if r.boxes is None or r.boxes.xyxy is None:
            continue

        xyxy = r.boxes.xyxy.cpu().numpy()
        cls = r.boxes.cls.cpu().numpy().astype(int)
        scr = r.boxes.conf.cpu().numpy()

        for (x1, y1, x2, y2), c, s in zip(xyxy, cls, scr):
            if c not in to_bdd_cat:
                continue
            cat_id = to_bdd_cat[c]
            cname = BDD10_NAMES[cat_id - 1]
            if s < CLASS_CONF.get(cname, 0.20):
                continue
            w, h = max(0.0, x2 - x1), max(0.0, y2 - y1)
            results.append({
                "image_id": img_id,
                "category_id": cat_id,
                "bbox": [float(x1), float(y1), float(w), float(h)],
                "score": float(s)
            })

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f)
    print(f"COCO-style detections => {out_json}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--images_dir", required=True)
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--imgsz", type=int, default=1536)
    ap.add_argument("--device", default=None)
    ap.add_argument("--tta", type=int, default=1)
    args = ap.parse_args()
    infer(**vars(args))
