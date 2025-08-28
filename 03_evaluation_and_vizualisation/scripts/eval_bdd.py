# scripts/eval_bdd.py
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import json
from pathlib import Path
import numpy as np
import argparse
s

def load_and_repair_gt(gt_path):
    """Load GT JSON and ensure 'info' and
    'licenses' exist so COCO doesn't crash."""
    gt_path = Path(gt_path)
    data = json.loads(gt_path.read_text(encoding="utf-8"))
    if "info" not in data:
        data["info"] = {
            "description":
                "BDD100K det to COCO (auto-patched)",
            "version": "1.0"}
    if "licenses" not in data:
        data["licenses"] = []
    # Build COCO object from dict
    #  (avoids re-serializing to disk)
    coco_gt = COCO()
    coco_gt.dataset = data
    coco_gt.createIndex()
    return coco_gt


def load_dt(coco_gt, dt_path_or_list):
    """Load detections. Accept a JSON path
      or an in-memory list of dicts."""
    if isinstance(dt_path_or_list, (str, Path)):
        dt = json.loads(Path(dt_path_or_list).read_text(encoding="utf-8"))
    else:
        dt = dt_path_or_list
    return coco_gt.loadRes(dt), dt


def _summary_dict(ev):
    """COCOeval.stats → named dict."""
    names = [
        "AP", "AP50", "AP75", "AP_small", "AP_medium", "AP_large",
        "AR@1", "AR@10", "AR@100", "AR_small", "AR_medium", "AR_large",
    ]
    vals = (ev.stats.tolist() if
            getattr(ev, "stats", None) is not None else [])
    return {names[i]: float(vals[i])
            for i in range(min(len(names), len(vals)))}


def _per_class_aps(ev, coco_gt):
    """Per-class AP for [.50:.95] and AP@0.50."""
    precisions = ev.eval["precision"]  # [T, R, K, A, M]
    T, R, K, A, M = precisions.shape
    cat_ids = coco_gt.getCatIds()
    cats = coco_gt.loadCats(cat_ids)
    id2name = {c["id"]: c["name"] for c in cats}

    # Index for IoU=0.50
    iou_thrs = ev.params.iouThrs
    idx50 = int(np.argmin(np.abs(iou_thrs - 0.50)))

    per_class = []
    for k, cid in enumerate(cat_ids):
        # [.50:.95]
        ps_all = precisions[:, :, k, 0, -1]
        ps_all = ps_all[ps_all > -1]
        ap_all = float(np.mean(ps_all)) if ps_all.size else float("nan")
        # @0.50
        ps_50 = precisions[idx50:idx50+1, :, k, 0, -1]
        ps_50 = ps_50[ps_50 > -1]
        ap_50 = float(np.mean(ps_50)) if ps_50.size else float("nan")

        per_class.append({
            "category_id": int(cid),
            "category_name": id2name[cid],
            "AP": ap_all,
            "AP50": ap_50,
        })
    return per_class


def coco_eval(gt_json, dt_json, iou_type="bbox"):
    coco_gt = load_and_repair_gt(gt_json)
    coco_dt, dt_list = load_dt(coco_gt, dt_json)

    # Optional sanity check: warn if image_ids don't overlap much
    gt_ids = set(coco_gt.getImgIds())
    if isinstance(dt_json, (str, Path)):
        dt_list = json.loads(Path(dt_json).read_text(encoding="utf-8"))
    dt_ids = {int(d["image_id"]) for d in (dt_list or [])}
    overlap = len(gt_ids & dt_ids)
    if dt_ids and overlap == 0:
        print("[WARN] No overlap between GT image_ids and DT image_ids. "
              "Ensure your inference used the same image_id mapping as GT.")

    ev = COCOeval(coco_gt, coco_dt, iouType=iou_type)
    ev.evaluate()
    ev.accumulate()
    ev.summarize()
    return ev, coco_gt, dt_list


def save_results_json(out_path, ev, coco_gt, dt_list, iou_type):
    out = {
        "iouType": iou_type,
        "summary": _summary_dict(ev),
        "per_class": _per_class_aps(ev, coco_gt),
        "params": {
            "iouThrs": [float(x) for x in ev.params.iouThrs.tolist()],
            "recThrs": [float(x) for x in ev.params.recThrs.tolist()],
            "maxDets": [int(x) for x in ev.params.maxDets],
            "areaRngLbl": list(ev.params.areaRngLbl),
        },
        "counts": {
            "num_gt_images": int(len(coco_gt.getImgIds())),
            "num_gt_annotations": int(len(coco_gt.anns)),
            "num_dt_detections": int(len(dt_list or [])),
        },
    }
    Path(out_path).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"[INFO] Saved metrics → {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt_json", required=True,
                    help="COCO-format ground truth JSON")
    ap.add_argument("--dt_json", required=True,
                     help="Detections JSON (list of results)")
    ap.add_argument("--iou_type", default="bbox",
                     choices=["bbox", "segm"])
    ap.add_argument("--out_json", default=None,
                     help="Where to save metrics JSON "
                        "(default: <dt_json>.eval.json)")
    args = ap.parse_args()

    ev, coco_gt, dt_list = coco_eval(args.gt_json,
                                     args.dt_json,
                                     iou_type=args.iou_type)

    out_json = args.out_json or (str(Path(args.dt_json)
                                     .with_suffix("_eval.json")))
    save_results_json(out_json, ev, coco_gt, dt_list, args.iou_type)
