# train fresh
# python train.py --device 0 --workers 2 --img 640 --batch 4 --epochs 100 --data data/dataset.yaml --weights yolov5s.pt

# finetune
# python train.py --device 0 --workers 2 --img 640 --batch 4 --epochs 100 --data data/dataset.yaml --hyp data/hyps/hyp.finetune.vehicle.yaml --weights /workdir/yolov5/runs/train/exp/weights/best.pt

# resume
python train.py --device 0 --workers 2 --img 640 --batch 4 --epochs 200 --data data/dataset.yaml --weights /workdir/yolov5/runs/train/exp4/weights/last.pt --resume


# $ python train.py --data coco.yaml --cfg yolov5s.yaml --weights '' --batch-size 64
#                                          yolov5m                                40
#                                          yolov5l                                24
#                                          yolov5x                                16

# 从预训练权重开始。推荐用于中小型数据集（即 VOC、VisDrone、GlobalWheat）。将模型的名称传递给--weights参数。模型会从最新的 YOLOv5 版本自动下载。
# python train.py --data custom.yaml --weights yolov5s.pt
#                                              yolov5m.pt
#                                              yolov5l.pt
#                                              yolov5x.pt
#                                              custom_pretrained.pt
# 白手起家。推荐用于大型数据集（即 COCO、Objects365、OIv6）。传递您感兴趣的模型架构 yaml 以及一个空--weights ''参数：
# python train.py --data custom.yaml --weights ' ' --cfg yolov5s.yaml
#                                                       yolov5m.yaml
#                                                       yolov5l.yaml
#                                                       yolov5x.yaml

python train.py \
--device 0 \
--workers 2 \
--img 640 \
--batch 4 \
--epochs 200 \
--data data/dataset.yaml \
--cfg models/yolov5s.yaml \
--weights /workdir/yolov5/runs/train/exp4/weights/best.pt
