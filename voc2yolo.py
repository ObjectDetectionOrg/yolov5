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