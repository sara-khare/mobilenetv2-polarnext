# check_labels_with_coco.py
import json
from pathlib import Path
from pycocotools.coco import COCO

# Adjust these paths
train_json = Path("/home/khare/dataset/coco_reduced/annotations/instances_train2017_remap_clean.json")
val_json = Path("/home/khare/dataset/coco_reduced/annotations/instances_val2017_remap_clean.json")

# Your target class names in the order you want them mapped to 0..4
wanted_classes = ['person', 'bicycle', 'car', 'truck', 'bus']

def check_json(p: Path):
    print(f"\n=== Checking {p} ===")
    if not p.exists():
        print("  File not found:", p)
        return
    coco = COCO(str(p))
    # pycocotools COCO uses getCatIds(catNms=...)
    cat_ids_for_wanted = coco.getCatIds(catNms=wanted_classes)
    cat2label = {cat_id: i for i, cat_id in enumerate(cat_ids_for_wanted)}
    print("  Mapped COCO cat ids for wanted classes:", cat_ids_for_wanted)
    # gather unique category_ids in annotations
    ann_ids = coco.getAnnIds()
    anns = coco.loadAnns(ann_ids)
    all_cids = sorted({a['category_id'] for a in anns})
    print("  Unique category_ids present in json:", all_cids)
    # show which of those are unknown to wanted set
    unknown = [cid for cid in all_cids if cid not in cat2label]
    if unknown:
        print("  >>> UNKNOWN / unexpected category_ids (not in wanted classes):", unknown)
    else:
        print("  All category_ids are in the wanted classes.")
    # map present cids to labels where possible and show mapping
    mapped_labels = sorted({cat2label[cid] for cid in all_cids if cid in cat2label})
    print("  Mapped labels (0..N-1):", mapped_labels)

if __name__ == "__main__":
    check_json(train_json)
    check_json(val_json)
