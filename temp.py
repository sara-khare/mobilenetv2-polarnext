import json
data = json.load(open("/home/khare/dataset/coco_reduced/annotations/instances_train2017.json"))
print(data["annotations"][:5])
