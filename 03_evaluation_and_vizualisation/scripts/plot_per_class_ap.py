import json
from pathlib import Path
import matplotlib.pyplot as plt

def main(eval_json, out_png="per_class_ap.png"):
    data = json.loads(Path(eval_json).read_text(encoding="utf-8"))
    rows = data["per_class"]
    names = [r["category_name"] for r in rows]
    aps   = [r["AP"] * 100.0 for r in rows]

    plt.figure(figsize=(10,5))
    plt.bar(names, aps)
    plt.xticks(rotation=30, ha="right")
    plt.ylabel("AP @[.50:.95] (%)")
    plt.title("Per-class AP (BDD val)")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    print(f"[INFO] Saved {out_png}")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_json", required=True)
    ap.add_argument("--out_png", default="per_class_ap.png")
    args = ap.parse_args()
    main(args.eval_json, args.out_png)
