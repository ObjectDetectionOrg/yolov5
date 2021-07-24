[TOC]

https://github.com/ultralytics/yolov5

https://docs.ultralytics.com/



https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data



https://github.com/ultralytics/yolov5/releases



## 数据标注

labelimg

标注文件格式Pacal VOC



## 构建YOLO风格的标注数据

将pascal voc格式的标注文件转换为YOLO风格的标注数据。

### YOLO风格的标注数据

label为整数类型

bbox中心点相对于图片大小的比例坐标(x, y)，和bbox的width、height相对于图片大小的比例w, h

x, y, w, h为浮点数类型，范围(0.0, 1.0]

```
label x y w h
```



### voc2yolo.py

需要修改

  classes = ["vehicle", "other"]

  root_dir = "/workdir/datasets/vehicle_dataset/train"

  root_dir = "/workdir/datasets/vehicle_dataset/test"

```
import os
import xml.etree.ElementTree as ET

# box里保存的是ROI感兴趣区域的坐标（x，y的最大值和最小值）
# 返回值为ROI中心点相对于图片大小的比例坐标，和ROI的w、h相对于图片大小的比例
def convert(size, box):
    dw = 1./(size[0])
    dh = 1./(size[1])
    x = (box[0] + box[1])/2.0
    y = (box[2] + box[3])/2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x, y, w, h)
 
 
# 对于单个xml的处理
def convert_annotation(xml_file: str, classes: list):
    txt_file = xml_file.replace(".xml", ".txt")
 
    tree = ET.parse(xml_file)
    root = tree.getroot()
 
    size = root.find('size')
 
    w = int(size.find('width').text)
    h = int(size.find('height').text)
 
    # 在一个XML中每个Object的迭代
    for obj in root.iter('object'):
        # iter()方法可以递归遍历元素/树的所有子元素
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        # 如果训练标签中的品种不在程序预定品种，或者difficult = 1，跳过此object
        if cls not in classes or int(difficult) == 1:
            continue
        cls_id = classes.index(cls)#这里取索引，避免类别名是中文，之后运行yolo时要在cfg将索引与具体类别配对
        xmlbox = obj.find('bndbox')
 
        bbox = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(
            xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
        yolo_bbox = convert((w, h), bbox)

        with open(txt_file, "w") as f:
            f.write(str(cls_id) + " " + " ".join([str(a) for a in yolo_bbox]) + '\n')
 

if __name__ == "__main__":
    classes = ["vehicle", "other"]

    root_dir = "/workdir/datasets/vehicle_dataset/train"
    root_dir = "/workdir/datasets/vehicle_dataset/test"

    for filename in os.listdir(root_dir):
        if not filename.endswith(".xml"):
            continue
        filename = os.path.join(root_dir, filename)
        print(filename)
        convert_annotation(filename, classes)
    print("Finished")
```



### 构建YOLO数据集

建立软连接到train、test

```
vehicle_dataset_yolo/
|-- images
|   |-- test -> /workdir/datasets/vehicle_dataset/test
|   `-- train -> /workdir/datasets/vehicle_dataset/train
`-- labels
    |-- test -> /workdir/datasets/vehicle_dataset/test
    `-- train -> /workdir/datasets/vehicle_dataset/train
```



## 修改配置文件

yolov5/data/dataset.yaml

需要修改类别数、类别列表

```
# COCO 2017 dataset http://cocodataset.org - first 128 training images
# Train command: python train.py --data coco128.yaml
# Default dataset location is next to YOLOv5:
#   /parent
#     /datasets/coco128
#     /yolov5


# Train/val/test sets as 1) dir: path/to/imgs, 2) file: path/to/imgs.txt, or 3) list: [path/to/imgs1, path/to/imgs2, ..]
path: /workdir/datasets/vehicle_dataset_yolo  # dataset root dir
train: images/train  # train images (relative to 'path') 128 images
val: images/test  # val images (relative to 'path') 128 images
test:  # test images (optional)

# Classes
nc: 2  # number of classes
names: ['vehicle', 'other']  # class names


# Download script/URL (optional)
# download: https://github.com/ultralytics/yolov5/releases/download/v1.0/coco128.zip
```

## 模型训练

train.sh

```
python train.py --device 0 --workers 2 --img 640 --batch 4 --epochs 200 --data data/dataset.yaml --weights yolov5s.pt
```



## 模型评估

```
# usage: val.py [-h] [--data DATA] [--weights WEIGHTS [WEIGHTS ...]] [--batch-size BATCH_SIZE] [--imgsz IMGSZ] [--conf-thres CONF_THRES]
#               [--iou-thres IOU_THRES] [--task TASK] [--device DEVICE] [--single-cls] [--augment] [--verbose] [--save-txt]
#               [--save-hybrid] [--save-conf] [--save-json] [--project PROJECT] [--name NAME] [--exist-ok] [--half]

python val.py \
--data data/dataset.yaml \
--weights runs/train/exp4/weights/best.pt \
--batch-size 4 \
--imgsz 640 \
--conf-thres 0.5 \
--iou-thres 0.5 \
--device 0
```



## 模型测试

```
python test.py \
--data data/dataset.yaml \
--weights runs/train/exp4/weights/best.pt \
--batch-size 4 \
--img-size 640 \
--conf-thres 0.5 \
--iou-thres 0.5 \
--device 0 \
--half
```

**测试结果：**

```
test: data=data/dataset.yaml, weights=['runs/train/exp4/weights/best.pt'], batch_size=4, imgsz=640, conf_thres=0.5, iou_thres=0.5, task=val, device=0, single_cls=False, augment=False, verbose=False, save_txt=False, save_hybrid=False, save_conf=False, save_json=False, project=runs/test, name=exp, exist_ok=False, half=True
YOLOv5 🚀 v5.0-290-g62409ee torch 1.8.0 CUDA:0 (Tesla V100S-PCIE-32GB, 32510.5MB)

Fusing layers...
Model Summary: 224 layers, 7056607 parameters, 0 gradients, 16.3 GFLOPs
val: Scanning '/workdir/datasets/vehicle_dataset_yolo/labels/test.cache' images and labels... 414 found, 5 missing, 0 empty, 0 corrupted:
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 105/105 [00:03<00:00, 27.31it/s]
                 all        419        414      0.984      0.935      0.935      0.785
             vehicle        419        184      0.972      0.935       0.93      0.761
               other        419        230      0.995      0.935       0.94      0.809
Speed: 0.1ms pre-process, 2.0ms inference, 0.8ms NMS per image at shape (4, 3, 640, 640)
Results saved to runs/test/exp2
```



## 模型裁剪

Pruning model...  0.3 global sparsity

**mAP@0.5明显下降**

```
    # Prune
    from utils.torch_utils import prune
    prune(model, 0.3)
```

```
def prune(model, amount=0.3):
    # Prune model to requested global sparsity
    import torch.nn.utils.prune as prune
    print('Pruning model... ', end='')
    for name, m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            prune.l1_unstructured(m, name='weight', amount=amount)  # prune
            prune.remove(m, 'weight')  # make permanent
    print(' %.3g global sparsity' % sparsity(model))
```

```
test: data=data/dataset.yaml, weights=['runs/train/exp4/weights/best.pt'], batch_size=4, imgsz=640, conf_thres=0.5, iou_thres=0.5, task=val, device=0, single_cls=False, augment=False, verbose=False, save_txt=False, save_hybrid=False, save_conf=False, save_json=False, project=runs/test, name=exp, exist_ok=False, half=True
YOLOv5 🚀 v5.0-290-g62409ee torch 1.8.0 CUDA:0 (Tesla V100S-PCIE-32GB, 32510.5MB)

Fusing layers...
Model Summary: 224 layers, 7056607 parameters, 0 gradients, 16.3 GFLOPs
Pruning model...  0.3 global sparsity
val: Scanning '/workdir/datasets/vehicle_dataset_yolo/labels/test.cache' images and labels... 414 found, 5 missing, 0 empty, 0 corrupted:
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 105/105 [00:03<00:00, 26.86it/s]
                 all        419        414      0.987      0.828      0.829      0.672
             vehicle        419        184      0.975      0.842      0.841      0.638
               other        419        230          1      0.813      0.818      0.705
Speed: 0.1ms pre-process, 2.3ms inference, 0.7ms NMS per image at shape (4, 3, 640, 640)
Results saved to runs/test/exp3

```



## torch.hub模型推断demo

### hubconf.py

将hubconf.py复制到weights目录下

```
"""YOLOv5 PyTorch Hub models https://pytorch.org/hub/ultralytics_yolov5/

Usage:
    import torch
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
"""

import torch


def _create(name, pretrained=True, channels=3, classes=80, autoshape=True, verbose=True, device=None):
    """Creates a specified YOLOv5 model

    Arguments:
        name (str): name of model, i.e. 'yolov5s'
        pretrained (bool): load pretrained weights into the model
        channels (int): number of input channels
        classes (int): number of model classes
        autoshape (bool): apply YOLOv5 .autoshape() wrapper to model
        verbose (bool): print all information to screen
        device (str, torch.device, None): device to use for model parameters

    Returns:
        YOLOv5 pytorch model
    """
    from pathlib import Path

    from models.yolo import Model, attempt_load
    from utils.general import check_requirements, set_logging
    from utils.google_utils import attempt_download
    from utils.torch_utils import select_device

    file = Path(__file__).absolute()
    check_requirements(requirements=file.parent / 'requirements.txt', exclude=('tensorboard', 'thop', 'opencv-python'))
    set_logging(verbose=verbose)

    save_dir = Path('') if str(name).endswith('.pt') else file.parent
    path = (save_dir / name).with_suffix('.pt')  # checkpoint path
    try:
        device = select_device(('0' if torch.cuda.is_available() else 'cpu') if device is None else device)

        if pretrained and channels == 3 and classes == 80:
            model = attempt_load(path, map_location=device)  # download/load FP32 model
        else:
            cfg = list((Path(__file__).parent / 'models').rglob(f'{name}.yaml'))[0]  # model.yaml path
            model = Model(cfg, channels, classes)  # create model
            if pretrained:
                ckpt = torch.load(attempt_download(path), map_location=device)  # load
                msd = model.state_dict()  # model state_dict
                csd = ckpt['model'].float().state_dict()  # checkpoint state_dict as FP32
                csd = {k: v for k, v in csd.items() if msd[k].shape == v.shape}  # filter
                model.load_state_dict(csd, strict=False)  # load
                if len(ckpt['model'].names) == classes:
                    model.names = ckpt['model'].names  # set class names attribute
        if autoshape:
            model = model.autoshape()  # for file/URI/PIL/cv2/np inputs and NMS
        return model.to(device)

    except Exception as e:
        help_url = 'https://github.com/ultralytics/yolov5/issues/36'
        s = 'Cache may be out of date, try `force_reload=True`. See %s for help.' % help_url
        raise Exception(s) from e


def custom(path='path/to/model.pt', autoshape=True, verbose=True, device=None):
    # YOLOv5 custom or local model
    return _create(path, autoshape=autoshape, verbose=verbose, device=device)


def yolov5s(pretrained=True, channels=3, classes=80, autoshape=True, verbose=True, device=None):
    # YOLOv5-small model https://github.com/ultralytics/yolov5
    return _create('yolov5s', pretrained, channels, classes, autoshape, verbose, device)


def yolov5m(pretrained=True, channels=3, classes=80, autoshape=True, verbose=True, device=None):
    # YOLOv5-medium model https://github.com/ultralytics/yolov5
    return _create('yolov5m', pretrained, channels, classes, autoshape, verbose, device)


def yolov5l(pretrained=True, channels=3, classes=80, autoshape=True, verbose=True, device=None):
    # YOLOv5-large model https://github.com/ultralytics/yolov5
    return _create('yolov5l', pretrained, channels, classes, autoshape, verbose, device)


def yolov5x(pretrained=True, channels=3, classes=80, autoshape=True, verbose=True, device=None):
    # YOLOv5-xlarge model https://github.com/ultralytics/yolov5
    return _create('yolov5x', pretrained, channels, classes, autoshape, verbose, device)


def yolov5s6(pretrained=True, channels=3, classes=80, autoshape=True, verbose=True, device=None):
    # YOLOv5-small-P6 model https://github.com/ultralytics/yolov5
    return _create('yolov5s6', pretrained, channels, classes, autoshape, verbose, device)


def yolov5m6(pretrained=True, channels=3, classes=80, autoshape=True, verbose=True, device=None):
    # YOLOv5-medium-P6 model https://github.com/ultralytics/yolov5
    return _create('yolov5m6', pretrained, channels, classes, autoshape, verbose, device)


def yolov5l6(pretrained=True, channels=3, classes=80, autoshape=True, verbose=True, device=None):
    # YOLOv5-large-P6 model https://github.com/ultralytics/yolov5
    return _create('yolov5l6', pretrained, channels, classes, autoshape, verbose, device)


def yolov5x6(pretrained=True, channels=3, classes=80, autoshape=True, verbose=True, device=None):
    # YOLOv5-xlarge-P6 model https://github.com/ultralytics/yolov5
    return _create('yolov5x6', pretrained, channels, classes, autoshape, verbose, device)


if __name__ == '__main__':
    model = _create(name='yolov5s', pretrained=True, channels=3, classes=80, autoshape=True, verbose=True)  # pretrained
    # model = custom(path='path/to/model.pt')  # custom

    # Verify inference
    import cv2
    import numpy as np
    from PIL import Image

    imgs = ['data/images/zidane.jpg',  # filename
            'https://github.com/ultralytics/yolov5/releases/download/v1.0/zidane.jpg',  # URI
            cv2.imread('data/images/bus.jpg')[:, :, ::-1],  # OpenCV
            Image.open('data/images/bus.jpg'),  # PIL
            np.zeros((320, 640, 3))]  # numpy

    results = model(imgs)  # batched inference
    results.print()
    results.save()
```



### torchhub_inference.py

【详细文档】

https://github.com/ultralytics/yolov5/issues/36

https://docs.ultralytics.com/tutorials/pytorch-hub/



> Downloading https://github.com/ultralytics/yolov5/releases/download/v5.0/yolov5x.pt to /root/.cache/torch/hub/ultralytics_yolov5_master/yolov5x.pt



**--img-size 640**

| 模型       | pytorch推理时间 |
| ---------- | --------------- |
| yolov5s.pt | 11ms            |
| yolov5m.pt | 13ms            |
| yolov5x.pt | 18ms            |



```
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
```



## 模型部署

https://github.com/wang-xinyu/tensorrtx

https://github.com/wang-xinyu/tensorrtx/tree/master/yolov5



| 模型       | tensorrt推理时间 |
| ---------- | ---------------- |
| yolov5s.pt | 3ms              |
| yolov5m.pt |                  |
| yolov5x.pt |                  |



### 环境安装

通过dockerfile安装tensorrtx

```
docker build - < Dockerfile
```

临时打开镜像`nvidia-docker run -it hakuyyf/tensorrtx:trt7_cuda10 bash`

```
nvidia-docker run -itd --env HTTP_PROXY="http://10.18.47.108:3128" --env HTTPS_PROXY="https://10.18.47.108:3128" --env NO_PROXY="localhost,127.0.0.1,0.0.0.0" -p 9022:22 --name tensorrtx_zql -v /apps/zhongql3/workspace:/workdir -w /workdir hakuyyf/tensorrtx:trt7_cuda10 bash
```



### int8量化

模型减小；由于本身耗时较少，导致推理时间没有显著减少；精度降低。

| 模型       | 量化前的大小 | 量化后的大小 | tensorrt推理时间 |
| ---------- | ------------ | ------------ | ---------------- |
| yolov5s.pt | 19M          | 8.8M         | 3ms              |
| yolov5m.pt |              |              |                  |
| yolov5x.pt |              |              |                  |

