# save as create_coco_subset.py
import json
import random
import argparse
from pathlib import Path

def create_subset(ann_path, out_path, n_images, seed=42, shuffle=False):
    ann_path = Path(ann_path)
    out_path = Path(out_path)
    with ann_path.open('r') as f:
        coco = json.load(f)

    images = coco.get('images', [])
    anns = coco.get('annotations', [])
    cats = coco.get('categories', coco.get('categories', []))

    if shuffle:
        random.seed(seed)
        random.shuffle(images)

    if n_images <= 0 or n_images >= len(images):
        print(f"Requested n_images={n_images} -> writing full file ({len(images)} images).")
        selected_images = images
    else:
        selected_images = images[:n_images]
        print(f"Selected first {n_images} images out of {len(images)} total.")

    selected_ids = {img['id'] for img in selected_images}

    # Filter annotations to only those referring to selected image ids
    selected_anns = [a for a in anns if a.get('image_id') in selected_ids]

    out = {
        'info': coco.get('info', {}),
        'licenses': coco.get('licenses', None),
        'images': selected_images,
        'annotations': selected_anns,
        'categories': cats
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open('w') as f:
        json.dump(out, f)
    print(f"Wrote subset to: {out_path}  images={len(selected_images)} annotations={len(selected_anns)}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--in', dest='infile', required=True,
                        help='path to original coco val json')
    parser.add_argument('--out', dest='outfile', required=True,
                        help='path to output subset json')
    parser.add_argument('--n', type=int, default=500, help='number of images in subset')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--shuffle', action='store_true',
                        help='shuffle images before picking first n (useful to randomize)')
    args = parser.parse_args()

    create_subset(args.infile, args.outfile, args.n, seed=args.seed, shuffle=args.shuffle)
