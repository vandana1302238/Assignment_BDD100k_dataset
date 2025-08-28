"""Genrates bdd dataset to coco format
"""

import json
from pathlib import Path

# Default BDD10 classes
BDD10_NAMES = [
    "person", "rider", "car", "bus", "truck",
    "bike", "motor", "traffic light", "traffic sign", "train"
]
NAME2ID = {n: i+1 for i, n in enumerate(BDD10_NAMES)}


def bdd_to_coco_gt(bdd_json_path, images_root, out_json_path, attrs_out):
    bdd_json_path = Path(bdd_json_path)
    images_root = Path(images_root)
    out_json_path = Path(out_json_path)
    attrs_out = Path(attrs_out)

    with open(bdd_json_path, "r", encoding="utf-8") as f:
        bdd = json.load(f)

    images, annotations = [], []
    image_attrs = {}
    ann_id = 1

    for img_id, item in enumerate(bdd, start=1):
        fname = item["name"]
        images.append({
            "id": img_id,
            "file_name": fname,
            "width": 1280,
            "height": 720
        })

        attrs = item.get("attributes", {})
        image_attrs[img_id] = {
            "weather":   attrs.get("weather", "undefined"),
            "timeofday": attrs.get("timeofday", "undefined"),
            "scene":     attrs.get("scene", "undefined"),
        }

        for lab in item.get("labels", []):
            cat = lab.get("category")
            if cat not in NAME2ID:
                continue
            box = lab.get("box2d")
            if not box:
                continue
            x1, y1, x2, y2 = box["x1"], box["y1"], box["x2"], box["y2"]
            w, h = max(0.0, x2 - x1), max(0.0, y2 - y1)
            annotations.append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": NAME2ID[cat],
                "bbox": [x1, y1, w, h],
                "area": w*h,
                "iscrowd": 0
            })
            ann_id += 1

    categories = [{"id": cid, "name": name}
                  for name, cid in NAME2ID.items()]
    coco = {"images": images, "annotations": annotations,
             "categories": categories}

    out_json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(coco, f)
    with open(attrs_out, "w", encoding="utf-8") as f:
        json.dump(image_attrs, f)

    print(f"COCO GT => {out_json_path}")
    print(f"Image attributes => {attrs_out}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--bdd_json", required=True,
                    help=r'…\bdd100k_labels_images_{train|val}.json')
    ap.add_argument("--images_root", required=True,
                    help=r'…\images\100k\{train|val}')
    ap.add_argument("--out", required=True)
    ap.add_argument("--attrs_out", required=True)
    args = ap.parse_args()
    bdd_to_coco_gt(bdd_json_path=args.bdd_json,
        images_root=args.images_root,
        out_json_path=args.out,
        attrs_out=args.attrs_out)
