import json
from pathlib import Path
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def load_and_repair_gt(gt_path):
    """Load GT JSON and ensure 'info'
    and 'licenses' exist so COCO doesn't crash."""
    gt_path = Path(gt_path)
    data = json.loads(gt_path.read_text(
        encoding="utf-8"))
    # Patch missing required keys for
    # pycocotools loadRes
    data.setdefault("info", {"description":
                                 "GT auto-patched for COCOeval",
                             "version": "1.0"})
    data.setdefault("licenses", [])
    # Build COCO object from dict (avoid re-serializing)
    coco_gt = COCO()
    coco_gt.dataset = data
    coco_gt.createIndex()
    return coco_gt


def load_dt(coco_gt, dt_json_path):
    """Load detections from path
    (list of dicts) and filter to GT categories."""
    dt_list = json.loads(Path(dt_json_path
                              ).read_text(encoding="utf-8"))
    # Optional: filter out preds that
    # reference categories not in GT
    gt_cat_ids = set(coco_gt.getCatIds())
    dt_list = [d for d in dt_list
               if int(d.get("category_id", -1)) in gt_cat_ids]
    coco_dt = coco_gt.loadRes(dt_list)
    return coco_dt, dt_list


def per_class_ap(coco_gt, coco_dt):
    ev = COCOeval(coco_gt, coco_dt,
                  iouType="bbox")
    ev.evaluate()
    ev.accumulate()
    p = ev.eval["precision"]  # shape: [T, R, K, A, M]
    # list of category ids (aligned with K dim)
    cat_ids = ev.params.catIds
    area_rng_index = 0  # index for area='all'
    # last index corresponds to maxDet default (e.g., 100)
    m_idx = len(ev.params.maxDets) - 1

    ap_by_cat = {}
    for k, cid in enumerate(cat_ids):
        # precision slice for this category across IoUs and recalls
        pk = p[:, :, k, area_rng_index, m_idx]
        pk = pk[pk > -1]
        ap = float(pk.mean()) if pk.size else 0.0
        ap_by_cat[int(cid)] = ap
    return ap_by_cat, ev


def main(gt_json, dt_json, out_json):
    coco_gt = load_and_repair_gt(gt_json)
    coco_dt, dt_list = load_dt(coco_gt, dt_json)

    # Sanity: warn if image_id overlap is zero
    gt_ids = set(coco_gt.getImgIds())
    dt_ids = {int(d["image_id"])
              for d in (dt_list or [])}
    if dt_ids and len(gt_ids & dt_ids) == 0:
        print("[WARN] No overlap between GT "
              "image_ids and DT image_ids. "
              "Ensure your inference used "
              "the same image_id mapping as GT.")

    ap_by_cat, ev = per_class_ap(coco_gt, coco_dt)
    id_to_name = {c["id"]: c["name"]
                  for c in coco_gt.loadCats(
            coco_gt.getCatIds())}

    records = [
        {"category_id": cid, "category":
            id_to_name.get(cid, str(cid)),
         "AP": float(ap)}
        for cid, ap in ap_by_cat.items()
    ]
    records.sort(key=lambda r: r["category"])

    # JSON
    Path(out_json).parent.mkdir(parents=True,
                                exist_ok=True)
    Path(out_json).write_text(json.dumps(records,
                                         indent=2),
                              encoding="utf-8")

    print(f"[INFO] Per-class JSON → {out_json}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt_json", required=True)
    ap.add_argument("--dt_json", required=True)
    ap.add_argument("--out_json", required=True)
    args = ap.parse_args()
    main(**vars(args))
