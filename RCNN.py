import torch
import cv2
import torchvision
from PIL import Image
from torchvision.transforms import ToTensor
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights, FasterRCNN_ResNet50_FPN_V2_Weights
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
plt.rc('font',family='Times New Roman')
def savefig(image, boxes, labels, scores, path):
    fig, ax = plt.subplots()
    ax.imshow(image)
    for box, label, score in zip(boxes, labels, scores):
        if score > 0.5:
            rect = patches.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], linewidth=1, edgecolor='r',
                                     facecolor='none')
            ax.add_patch(rect)
            plt.text(box[0], box[1], f'{label}: {score}', color='white', bbox=dict(facecolor='red', alpha=0.5))

    #plt.savefig(path)

    return

model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights='COCO_V1')
model.eval()
path_list = os.listdir('D:gallery')
features = []
for root, dirs, files in os.walk('D:gallery'):
    for file in sorted(path_list,key=lambda x:int(x[:-4])):
        print(os.path.join(root, file))
        image = Image.open(os.path.join(root, file)).convert("RGB")
        image_tensor = ToTensor()(image)
        prediction = model([image_tensor])
        boxes = prediction[0]["boxes"]
        labels = prediction[0]["labels"]
        scores = prediction[0]["scores"]
        boxes = boxes.detach().numpy()
        labels = labels.detach().numpy()
        scores = scores.detach().numpy()
        max_index = np.argsort(scores)[::-1][:3]
        max_box = boxes[max_index]
        img = np.average(np.array(image),axis=-1)
        temp = []
        for box in max_box:
            box = np.array(box,dtype=int)
            try:
                feature = np.array(cv2.resize(img[box[0]:box[2], box[1]:box[3]],dsize=[64,64])).reshape(-1)
                temp.append(feature)
            except Exception as e:
                print(str(e))
        features.append(temp)
import pickle

with open('Gallery_fea_full,'wb+') as f:
    pickle.dump(features,f)
