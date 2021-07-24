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