import onnxruntime
import numpy as np
from PIL import Image
import time
import matplotlib.pyplot as plt


# TODO: complete nms
def onnxruntime_inference():
    # 4 (1, 25200, 85)
    onnx_model_name = "yolov5/yolov5s.onnx"
    filename = "yolov5/images/zidane.jpg"

    # 4 (1, 25200, 7)
    # onnx_model_name = "/workdir/yolov5/runs/train/exp4/weights/best.onnx"
    # filename = "/workdir/yolov5/images/vehicleL.jpg"

    with open(filename, "rb") as f:
        img = Image.open(f)
        h = img.height
        w = img.width
        img = img.convert("RGB").resize((640, 640))
        plt.imshow(img)
        plt.show()

    # mean = np.array([0.485, 0.456, 0.406]).astype('float32')
    # stddev = np.array([0.229, 0.224, 0.225]).astype('float32')
    # tmpImg = (np.asarray(img).astype('float32') / float(255.0) - mean) / stddev
    # # change the r,g,b to b,r,g from [0,255] to [0,1]
    # # transforms.Normalize(mean = (0.485, 0.456, 0.406), std = (0.229, 0.224, 0.225))
    # tmpImg = tmpImg.transpose((2, 0, 1))
    # tmpImg = tmpImg[np.newaxis, :, :, :]

    tmpImg = np.asanyarray(img).transpose((2, 0, 1)).astype('float32')
    tmpImg = tmpImg[np.newaxis, :, :, :]

    ort_session = onnxruntime.InferenceSession(onnx_model_name)

    # # 1 x 3 x 1068 x 800
    # tmpImg = np.asanyarray(img)
    # tmpImg = np.reshape(tmpImg, (3, 1067, 800))
    # # compute ONNX Runtime output prediction
    ort_inputs = {ort_session.get_inputs()[0].name: tmpImg}
    pred = ort_session.run(None, ort_inputs)
    print(pred)
    print(len(pred), np.shape(pred[0]))

    # predict = pred.squeeze()
    # predict_np = (predict * 255).astype("uint8")
    #
    # im = Image.fromarray(predict_np).convert('RGB')
    #
    # imo = im.resize((w, h), resample=Image.BILINEAR)
    # imo.save("result.jpg")
    #
    # plt.imshow(imo, cmap='gray')
    # plt.show()


if __name__ == '__main__':
    onnxruntime_inference()
