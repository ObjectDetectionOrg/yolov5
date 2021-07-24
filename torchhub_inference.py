import base64
import cv2
from io import BytesIO
from PIL import Image
import os
import time
import torch

# Model
# model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # or yolov5m, yolov5x, custom
# model = torch.hub.load('ultralytics/yolov5', 'yolov5x')  # or yolov5m, yolov5x, custom

# path = '/workdir/yolov5'
# model = torch.hub.load(path, 'yolov5s', source="local")

path = '/workdir/yolov5/runs/train/exp4/weights'
# model = torch.hub.load(path, 'yolov5s', source="local")
# model = torch.hub.load('ultralytics/yolov5', 'custom', path='path/to/best.pt')  # default
# model = torch.hub.load('path/to/yolov5', 'custom', path='path/to/best.pt', source='local')  # local repo
model = torch.hub.load(path, 'custom', path='{}/best.pt'.format(path), source='local')  # local repo


# filename = "{}/images/zidane.jpg".format(os.path.dirname(__file__))
filename = "{}/images/vehicleL.jpg".format(os.path.dirname(__file__))
# filename = "{}/images/other.jpg".format(os.path.dirname(__file__))

# Images
# img can be file, PIL, OpenCV, numpy, multiple
img = cv2.imread(filename)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Inference
num = 10
for i in range(num):
    t = time.time()
    results = model(img)
    t = time.time() - t
    print("time: {:.2f}".format(t * 1000))

# Results
results.print()  # or .show(), .save(), .crop(), .pandas(), etc.
results.save(os.path.join(os.path.dirname(__file__), "output"))

# base64 results
results.imgs # array of original images (as np array) passed to model for inference
results.render()  # updates results.imgs with boxes and labels
for img in results.imgs:
    buffered = BytesIO()
    img_base64 = Image.fromarray(img)
    img_base64.save(buffered, format="JPEG")
    print(base64.b64encode(buffered.getvalue()).decode('utf-8'))  # base64 encoded image with results

# json results
json_result = results.pandas().xyxy[0].to_json(orient="records")  # JSON img1 predictions
print(json_result)
