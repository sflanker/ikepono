import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import matplotlib.pyplot as plt
import cv2
import numpy as np
import random


# load a model pre-trained on COCO
model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights=True)

model.eval();

from torchvision.io import read_image
img_path = "/mnt/d/scratch_data/mantas/by_name/original_2023/kona/Akari/Akari_20191227_01.jpg"
img = read_image(img_path)
img

img_f = img / 255.0
img_f

pred = model([img_f])
pred

COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]
pred_class = [COCO_INSTANCE_CATEGORY_NAMES[i] for i in list(pred[0]['labels'].numpy())]
pred_class

threshold = 0.25
pred_score = list(pred[0]['scores'].detach().numpy())
pred_f = [pred_score.index(x) for x in pred_score if x>threshold]
pred_t = pred_f[-1]
masks = (pred[0]['masks']>0.5).squeeze().detach().cpu().numpy()
pred_class = [COCO_INSTANCE_CATEGORY_NAMES[i] for i in list(pred[0]['labels'].numpy())]
pred_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(pred[0]['boxes'].detach().numpy())]
masks = masks[:pred_t+1]
pred_boxes = pred_boxes[:pred_t+1]
pred_class = pred_class[:pred_t+1]


def random_colour_masks(image):
  colours = [[0, 255, 0],[0, 0, 255],[255, 0, 0],[0, 255, 255],[255, 255, 0],[255, 0, 255],[80, 70, 180],[250, 80, 190],[245, 145, 50],[70, 150, 250],[50, 190, 190]]
  r = np.zeros_like(image).astype(np.uint8)
  g = np.zeros_like(image).astype(np.uint8)
  b = np.zeros_like(image).astype(np.uint8)
  r[image == 1], g[image == 1], b[image == 1] = colours[random.randrange(0,10)]
  coloured_mask = np.stack([r, g, b], axis=2)
  return coloured_mask


boxes = pred_boxes
img = cv2.imread(img_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
for i in range(len(masks)):
    rgb_mask = random_colour_masks(masks[i])
    img = cv2.addWeighted(img, 1, rgb_mask, 0.5, 0)
    pt1 = [int(v) for v in boxes[i][0]]
    pt2 = [int(v) for v in boxes[i][1]]
    print(f"Pt1 = {pt1}, pt2 = {pt2}")
    cv2.rectangle(img, pt1, pt2,color=(0, 255, 0), thickness=1)
    cv2.putText(img,text = pred_class[i], org = pt1, fontFace = cv2.FONT_HERSHEY_SIMPLEX, fontScale = .75, color=(0,255,0),thickness=1)
plt.figure(figsize=(20,30))
plt.imshow(img)
plt.xticks([])
plt.yticks([])
plt.show()

# +
# cv2.putText?
# -


