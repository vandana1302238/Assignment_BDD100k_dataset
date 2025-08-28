# scripts/plot_pr_curve.py
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def load_and_repair_gt(gt_path):
    """Load GT JSON and ensure 'info' and 'licenses' exist so COCO doesn't crash."""
    gt_path = Path(gt_path)
    data = json.loads(gt_path.read_text(encoding="utf-8"))
    data.setdefault("info", {"description": "GT auto-patched for COCOeval", "version": "1.0"})
    data.setdefault("licenses", [])
    coco_gt = COCO()
    coco_gt.dataset = data
    coco_gt.createIndex()
    return coco_gt


def pr_curve(gt_json, dt_json, class_names, out_prefix="pr_curve"):
    cocoGt = load_and_repair_gt(gt_json)

    # Load detections and filter to GT categories (safer)
    dt_list = json.loads(Path(dt_json).read_text(encoding="utf-8"))
    gt_cat_ids = set(cocoGt.getCatIds())
    dt_list = [d for d in dt_list if int(d.get("category_id", -1)) in gt_cat_ids]
    cocoDt = cocoGt.loadRes(dt_list)

    # Evaluate once (bbox)
    ev = COCOeval(cocoGt, cocoDt, iouType="bbox")
    ev.evaluate()
    ev.accumulate()

    # Build name-id maps
    cats = cocoGt.loadCats(cocoGt.getCatIds())
    name2id = {c["name"]: c["id"] for c in cats}
    catIds  = ev.params.catIds

    # Pull precision tensor and axes
    precisions = ev.eval["precision"]  # [T, R, K, A, M]
    iouThrs   = ev.params.iouThrs
    recThrs   = ev.params.recThrs
    A_idx     = 0                      # area='all'
    M_idx     = -1                     # last maxDet
    i50_idx   = int(np.argmin(np.abs(iouThrs - 0.50)))

    # Plot per requested class
    for name in class_names:
        if name not in name2id:
            print(f"[WARN] class '{name}' not found in GT; skipping")
            continue
        cid = name2id[name]
        if cid not in catIds:
            print(f"[WARN] class id {cid} not in eval params; skipping")
            continue
        k = catIds.index(cid)

        pr = precisions[i50_idx, :, k, A_idx, M_idx]
        valid = pr > -1
        if not np.any(valid):
            print(f"[WARN] no valid PR data for '{name}'")
            continue

        plt.figure()
        plt.plot(recThrs[valid], pr[valid])
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title(f"PR @IoU=0.50 — {name}")
        out_png = f"{out_prefix}_{name.replace(' ', '_')}.png"
        plt.tight_layout()
        plt.savefig(out_png, dpi=200)
        plt.close()
        print(f"[INFO] Saved {out_png}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt_json", required=True)
    ap.add_argument("--dt_json", required=True)
    ap.add_argument("--eval_json", required=True, help="Used only to pick worst classes")
    ap.add_argument("--top_k", type=int, default=3)
    args = ap.parse_args()

    data = json.loads(Path(args.eval_json).read_text(encoding="utf-8"))
    worst = sorted(data["per_class"], key=lambda r: r["AP"])[:args.top_k]
    names = [w["category_name"] for w in worst]
    pr_curve(args.gt_json, args.dt_json, names)
