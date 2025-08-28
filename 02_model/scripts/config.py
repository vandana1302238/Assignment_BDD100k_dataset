# config.py
from pathlib import Path

# ROOTS 
IMAGES_ROOT = Path(r"C:\\Users\\DDB1COB\\Downloads\\assignment_data_bdd\\bdd100k_images_100k\\bdd100k\\images\\100k")
LABELS_ROOT = Path(r"C:\\Users\\DDB1COB\\Downloads\\assignment_data_bdd\\bdd100k_labels_release\\bdd100k\\labels")

# Split-specific
VAL_IMAGES_DIR  = IMAGES_ROOT / "val"
TRAIN_IMAGES_DIR = IMAGES_ROOT / "train"

VAL_BDD_JSON   = LABELS_ROOT / "bdd100k_labels_images_val.json"
TRAIN_BDD_JSON = LABELS_ROOT / "bdd100k_labels_images_train.json"

# Outputs 
OUT_DIR = Path(r"C:\\00_etl3\\11_assg\\00_solution\\outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

VAL_GT_COCO   = OUT_DIR / "val_gt_coco.json"
VAL_ATTRS     = OUT_DIR / "val_image_attrs.json"
VAL_DT_COCO   = OUT_DIR / "val_dt_coco.json"
VIZ_VAL_DIR   = OUT_DIR / "viz_val"

TRAIN_GT_COCO = OUT_DIR / "train_gt_coco.json"
TRAIN_ATTRS   = OUT_DIR / "train_image_attrs.json"
