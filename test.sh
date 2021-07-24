# python test.py --weights yolov5x.pt --data coco.yaml --img 640 --half

python test.py \
--data data/dataset.yaml \
--weights runs/train/exp4/weights/best.pt \
--batch-size 4 \
--img-size 640 \
--conf-thres 0.5 \
--iou-thres 0.5 \
--device 0 \
--half


# Prune
# python test_with_prune.py \
# --data data/dataset.yaml \
# --weights runs/train/exp4/weights/best.pt \
# --batch-size 4 \
# --img-size 640 \
# --conf-thres 0.5 \
# --iou-thres 0.5 \
# --device 0 \
# --half