import json
p = "/home/khare/dataset/coco_reduced/annotations/instances_train2017_remap_clean.json"
with open(p, "r") as f:
    d = json.load(f)
ids = set(ann['category_id'] for ann in d.get('annotations', []))
print("unique category ids in json:", sorted(ids))
# Now check what CocoPolarDataset will map:
from projects.PolarNeXt.model.coco import CocoPolarDataset  # adjust import path if needed
ds = CocoPolarDataset(ann_file=p, data_prefix=dict(img='/home/khare/dataset/coco_reduced/train2017'))
print("cat_ids:", ds.cat_ids)
print("cat2label:", ds.cat2label)
