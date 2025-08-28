import json
from pathlib import Path
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from datetime import datetime
import io
from contextlib import redirect_stdout
import argparse


def subset_ids(attrs_json, allowed):
    attrs = json.loads(Path(attrs_json).read_text(encoding="utf-8"))
    return {int(i) for i, a in attrs.items()
            if a.get("weather", "undefined") in allowed}


def build_slice_gt(gt_full, keep_ids):
    """Return a COCO GT dict for the slice,
     repairing 'info'/'licenses' if missing."""
    images = [im for im in gt_full.get("images", [])
              if int(im["id"]) in keep_ids]
    valid = {int(im["id"]) for im in images}
    anns = [a for a in gt_full.get("annotations", [])
            if int(a["image_id"]) in valid]

    gt_sub = {
        "info": gt_full.get("info", {"description"
                                     : "BDD100K slice "
                                       "(auto-patched)",
                                     "version": "1.0"}),
        "licenses": gt_full.get("licenses", []),
        "images": images,
        "annotations": anns,
        "categories": gt_full.get("categories", []),
    }
    return gt_sub, valid


def filter_dt(dt_list, valid_img_ids):
    return [d for d in dt_list
            if int(d.get("image_id", -1))
            in valid_img_ids]


def stats_to_dict(ev):
    names = ["AP", "AP50", "AP75", "APs", "APm",
             "APl", "AR@1", "AR@10", "AR@100",
             "ARs", "ARm", "ARl"]
    vals = getattr(ev, "stats", None)
    if vals is None:
        vals = [None] * 12
    else:
        vals = vals.tolist() \
            if hasattr(vals, "tolist") else list(vals)
    return {k: (None if v is None else float(v))
            for k, v in zip(names, vals)}


def eval_slice(gt_sub, dt_sub):
    if not gt_sub["images"]:
        print("  [skip] No images in this slice.")
        return None, None, None

    coco_gt = COCO()
    coco_gt.dataset = gt_sub
    coco_gt.createIndex()

    coco_dt = coco_gt.loadRes(dt_sub
                              if isinstance(dt_sub,
                                            list) else [])

    # sanity check for ID overlap
    gt_ids = set(coco_gt.getImgIds())
    dt_ids = {int(d["image_id"]) for d in dt_sub} if dt_sub else set()
    if dt_ids and len(gt_ids & dt_ids) == 0:
        print("  [WARN] No overlap between GT"
              " and DT image_ids in this slice. "
              "Did inference use the GT filename-image_id map?")

    ev = COCOeval(coco_gt, coco_dt, iouType="bbox")
    ev.evaluate()
    ev.accumulate()

    # capture printed summary AND echo it
    buf = io.StringIO()
    with redirect_stdout(buf):
        ev.summarize()
    summary_text = buf.getvalue()
    print(summary_text, end="")

    return ev, coco_gt, summary_text


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt_json", required=True)
    ap.add_argument("--dt_json", required=True)
    ap.add_argument("--attrs_json", required=True)
    ap.add_argument("--out_json", default=None,
                    help="Path to write JSON with "
                         "metrics per weather slice")
    args = ap.parse_args()

    gt_full = json.loads(Path(args.gt_json)
                         .read_text(encoding="utf-8"))
    dt_list = json.loads(Path(args.dt_json)
                         .read_text(encoding="utf-8"))

    slices = {
        "clear": {"clear"},
        "overcast": {"overcast", "partly cloudy"},
        "rainy": {"rainy"},
        "snowy": {"snowy"},
        "foggy": {"foggy"},
        "all": {"clear", "overcast", "partly cloudy",
                "rainy", "snowy", "foggy", "undefined"},
    }

    out_payload = {"timestamp": datetime.now().isoformat(
        timespec="seconds")}
    for name, allowed in slices.items():
        keep = subset_ids(args.attrs_json, allowed)
        if not keep:
            print(f"[{name}] no images")
            continue

        gt_sub, valid_ids = build_slice_gt(gt_full, keep)
        dt_sub = filter_dt(dt_list, valid_ids)

        print(f"\n=== Weather: {name} (N={len(gt_sub['images'])}) ===")
        ev, coco_gt, summary_text = eval_slice(gt_sub, dt_sub)
        if ev is None:
            continue

        out_payload[name] = {
            **stats_to_dict(ev),
            "num_images": len(gt_sub["images"]),
            "counts": {
                "images": len(coco_gt.getImgIds()),
                "annotations": len(coco_gt.getAnnIds()),
                "detections": len(dt_sub),
            },
            "coco_summarize": summary_text.strip(),
        }

    if args.out_json and out_payload:
        out_path = Path(args.out_json)
        out_path.parent.mkdir(parents=True,
                              exist_ok=True)
        out_path.write_text(json.dumps(out_payload,
                                       indent=2),
                            encoding="utf-8")
        print(f"[INFO] Per-weather metrics "
              f"JSON written - {out_path}")
