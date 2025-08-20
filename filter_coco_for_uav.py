import json
import os
import shutil

# --- User parameters ---
coco_train_json = "/home/khare/dataset/coco/annotations/instances_train2017.json"
coco_val_json   = "/home/khare/dataset/coco/annotations/instances_val2017.json"
train_images_root = "/home/khare/dataset/coco/train2017/"
val_images_root   = "/home/khare/dataset/coco/val2017/"
output_root       = "/home/khare/dataset/coco_reduced/"
selected_classes = ["person", "car", "truck", "bus", "bicycle"]

def reduce_coco(coco_json_path, images_root, output_images_dir, output_json_path):
    with open(coco_json_path) as f:
        coco = json.load(f)

    # Map category id → name
    catid2name = {cat["id"]: cat["name"] for cat in coco["categories"]}
    selected_ids = [cid for cid, name in catid2name.items() if name in selected_classes]

    # Map image id → image info
    imgid2info = {img["id"]: img for img in coco["images"]}

    # Filter annotations and collect images to keep
    filtered_annotations = []
    images_to_keep = set()
    for ann in coco["annotations"]:
        if ann["category_id"] in selected_ids:
            filtered_annotations.append(ann)
            images_to_keep.add(ann["image_id"])

    # Filter images info
    filtered_images = [imgid2info[iid] for iid in images_to_keep]

    print(f"{len(filtered_images)} images with {len(filtered_annotations)} annotations selected from {coco_json_path}")

    # Make output folder for images
    os.makedirs(output_images_dir, exist_ok=True)

    # Copy images
    for img_info in filtered_images:
        src = os.path.join(images_root, img_info["file_name"])
        dst = os.path.join(output_images_dir, img_info["file_name"])
        shutil.copy2(src, dst)

    # Prepare reduced COCO JSON
    reduced_coco = {
        "images": filtered_images,
        "annotations": filtered_annotations,
        "categories": [cat for cat in coco["categories"] if cat["id"] in selected_ids]
    }

    # Save reduced JSON
    with open(output_json_path, "w") as f:
        json.dump(reduced_coco, f)
    print(f"Saved reduced COCO JSON to {output_json_path}")

# --- Run for train and val ---
reduce_coco(
    coco_train_json,
    train_images_root,
    os.path.join(output_root, "train2017"),
    os.path.join(output_root, "annotations/instances_train2017.json")
)

reduce_coco(
    coco_val_json,
    val_images_root,
    os.path.join(output_root, "val2017"),
    os.path.join(output_root, "annotations/instances_val2017.json")
)
