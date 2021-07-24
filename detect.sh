# python detect.py --source images/zidane.jpg  # image 

# python detect.py \
# --img-size 640 \
# --source images/vehicleL.jpg \
# --weights /workdir/yolov5/runs/train/exp4/weights/yolov5s.pt \
# --device 0 \

python detect.py \
--img-size 1280 \
--source images/vehicleL.jpg \
--weights /workdir/yolov5/runs/train/exp4/weights/yolov5s.pt \
--device 0 \
--half