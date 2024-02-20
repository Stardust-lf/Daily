import torch
import cv2
import os
import glob
import numpy as np

if __name__ == '__main__':
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True, _verbose=False)
    results = []
    for root, dirs, files in os.walk('query'):
        for file in files:
            print(os.path.join(root, file))
            img = cv2.imread(os.path.join(root, file))[..., ::-1]
            result = model(img)
            results.append(result)
            print(result.pandas().xyxy[0])
            #result.save()
    # Inference
    #
    # # Results
    # results.print()
    # # results.save()  # or .show()
    # results.show()
    # print(results.xyxy[0])  # img1 predictions (tensor)
    # print(results.pandas().xyxy[0])  # img1 predictions