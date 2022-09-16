import os
import pickle
import xml.etree.ElementTree as ET
from tqdm import tqdm

# chance your classes here
names: ['crack','finger','black_core','thick_line','star_crack','corner','fragment','scratch',
        'horizontal_dislocation','vertical_dislocation','printing_error','short_circuit' ]

# function:
#       (xmin, xmax, ymin, ymax) -> (center_x, cneter_y, box_weight, box_height)
# size: (w, h)
# box : (xmin, xmax, ymin, ymax)
def convert(size, box):
    dw = 1.0 / size[0]
    dh = 1.0 / size[1]
    x = (box[0] + box[1]) / 2.0     # center_x
    y = (box[2] + box[3]) / 2.0     # center_y
    w = box[1] - box[0]             # box_weight
    h = box[3] - box[2]             # box_height
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)

# 功能：对单个文件进行转换，其中名字是相匹配的
# infile: xml文件路径
# outfile：txt文件输出路径
def convert_annotation(infile, outfile):
    in_file = open(infile, encoding='utf-8')  # xml path
    out_file = open(outfile, 'w')

    tree = ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult) == 1:   # 去掉无类别与过于困难的样本
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text),
             float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
        bb = convert((w, h), b)
        out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')

    in_file.close()
    out_file.close()

# 功能：对xml目录下的全部xml文件全部转换成yolo的txt格式，并输出在savepath目下
# xmlpath: the dir path save xml files
# savepath: the dir path to save txt files
def xml_to_txt(xmlpath, savepath):
    assert os.path.exists(xmlpath), "{} not exists.".format(xmlpath)
    if not os.path.exists(savepath):
        os.mkdir(savepath)
        assert os.path.exists(savepath), "{} create fail.".format(savepath)
    xmlfiles = os.listdir(xmlpath)
    assert len(xmlfiles) != 0, "{} have no files".format(xmlpath)
    xmlfilepaths = [os.path.join(xmlpath, xmlfile) for xmlfile in xmlfiles]
    for xmlfilepath in tqdm(xmlfilepaths, desc="change xml to txt: "):
        # E:\学习\机器学习\数据集\众盈比赛数据集\xml\train1.xml -> train1
        image_id = xmlfilepath.split('\\')[-1].split('.')[0]
        # train1 -> \..\train1.txt
        txtfilepath = os.path.join(savepath, image_id) + ".txt"
        convert_annotation(xmlfilepath, txtfilepath)


if __name__ == '__main__':

    xmlpath = r"E:\学习\机器学习\数据集\VOC2012\VOCdevkit\VOC2012\Annotations"
    savepath = r"E:\学习\机器学习\数据集\VOC2012\VOCdevkit\VOC2012\YoloLabels"
    xml_to_txt(xmlpath, savepath)

