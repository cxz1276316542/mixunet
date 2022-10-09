import xml.etree.ElementTree as ET
import os
from os import listdir, getcwd
from os.path import join
from PIL import Image

# 读取xml文件
def voc_xml_extract(xml_fpath, txt_fpath, classes):
    # 一次读入xml的ElementTree
    with open(xml_fpath,encoding = 'utf-8') as f:
        tree = ET.parse(f)
        root = tree.getroot()
        size = root.find('size')
        w = int(size.find('width').text)
        h = int(size.find('height').text)

    # 循环的将标记目标存入输出文件
    with open(txt_fpath, 'w') as f:
        for obj in root.iter('object'):
            difficult = obj.find('difficult').text
            clsname = obj.find('name').text
            if clsname not in classes or int(difficult) == 1:
                continue
            cls_id = classes.index(clsname)
            xmlbox = obj.find('bndbox')
            b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(
                xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
            bb = (b[0], b[2], b[1], b[3])
            f.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')
    return True


classes = ["benign", "malignancy"]
# 1 源路径
## 1.1 xml路径
source_xml = 'D:\\thyroid\\dataset\\VOC2007_3.05\\anno_malignancy\\'
## 1.2 img路径
source_img = 'D:\\thyroid\\dataset\\VOC2007_3.05\\e\\'
# 2 保存路径
## 2.1 voc txt 路径
voc_txt = './malignancyvoctxt/'
if not os.path.exists(voc_txt):
    os.makedirs(voc_txt)
# 3 目标路径
target = 'D:\\thyroid\\dataset\\VOC2007_3.05\\target\\'

if not os.path.exists(target):
    os.makedirs(target)



# -- 获取所有类别的列表
# 获取文件清单
filelist = [x for x in os.listdir(source_xml) if x.endswith('.xml')]

for xml in filelist:
    xmlfpath =  source_xml + xml
    voc_txtfpath = voc_txt + xml.replace('.xml', '.txt')
    voc_xml_extract(xmlfpath, voc_txtfpath, classes=classes)

# txt文件列表
txtlist = [x for x in os.listdir(voc_txt) if x.endswith('.txt')]

lines2 = []
for txt in txtlist:
    txt_file = voc_txt + txt
    # 读取txt文件
    with open(txt_file, 'r') as f:
        lines = f.readlines()
    lines1 = [line.replace('\n', '').split() for line in lines]
    # lines2 = []
    for line in lines1:
        classname = classes[int(line[0])]
        print(line[1:])
        xywh = [int(float(x)) for x in line[1:]]
        tem_res = [classname, xywh, txt.replace('.txt', '.jpg')]
        lines2.append(tem_res)

# left：与左边界的距离
# up：与上边界的距离
# right：还是与左边界的距离
# below：还是与上边界的距离
# 简而言之就是，左上右下。
import pandas as pd
df1 = pd.DataFrame(lines2, columns =['cate', 'coordinate', 'filename'])
df1['ord'] = df1.groupby(['cate']).cumcount() + 1
df1.sort_values(['cate', 'ord'])
for i in range(len(df1)):
    tem_dict = dict(df1.iloc[i])
    # 根据序号存图片名称
    pic_name = str(tem_dict['ord'])+'.jpg'
    source_img_fpath = source_img + tem_dict['filename']
    save_path = target + tem_dict['cate'] + '/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    # 读取图片，转为灰度图后存储
    with open(source_img_fpath ,'rb') as f :
        img = Image.open(f)
        img_crop = img.crop(tem_dict['coordinate'])
        img_crop.save(save_path + pic_name)


