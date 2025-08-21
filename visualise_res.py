# visualize_coco_results.py
import argparse, json, os
from collections import defaultdict
import cv2
from pycocotools.coco import COCO

def draw_boxes(img, boxes, labels, scores, catid2name):
    for box, lab, sc in zip(boxes, labels, scores):
        x,y,w,h = map(int, box)
        x2, y2 = x+w, y+h
        cv2.rectangle(img, (x,y), (x2,y2), (0,255,0), 2)
        text = f"{catid2name.get(lab,str(lab))}: {sc:.2f}"
        cv2.putText(img, text, (x, max(y-6,0)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
    return img

def main(args):
    coco = COCO(args.ann)
    ann_map = {c['id']: c['name'] for c in coco.dataset['categories']}
    preds = json.load(open(args.results))
    # group by image_id
    by_img = defaultdict(list)
    for p in preds:
        by_img[p['image_id']].append(p)

    os.makedirs(args.out_dir, exist_ok=True)
    img_ids = sorted(by_img.keys())[args.start: args.start + args.max_images]

    for img_id in img_ids:
        info = coco.loadImgs([img_id])[0]
        img_path = os.path.join(args.images, info['file_name'])
        if not os.path.exists(img_path):
            print("missing image:", img_path); continue
        img = cv2.imread(img_path)
        entries = sorted(by_img[img_id], key=lambda x: x['score'], reverse=True)[:args.topk]
        boxes = [e['bbox'] for e in entries]
        labels = [e['category_id'] for e in entries]
        scores = [e['score'] for e in entries]
        vis = draw_boxes(img, boxes, labels, scores, ann_map)
        outp = os.path.join(args.out_dir, info['file_name'])
        cv2.imwrite(outp, vis)
        print("wrote", outp)

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--ann', required=True, help='val annotation json (COCO)')
    p.add_argument('--results', required=True, help='results bbox json')
    p.add_argument('--images', required=True, help='path to val images root')
    p.add_argument('--out-dir', default='vis_out', help='where to save images')
    p.add_argument('--topk', type=int, default=10, help='top K boxes per image')
    p.add_argument('--start', type=int, default=0)
    p.add_argument('--max-images', type=int, default=50)
    args = p.parse_args()
    main(args)


