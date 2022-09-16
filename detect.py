# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Run inference on images, videos, directories, streams, etc.

Usage - sources:
    $ python path/to/detect.py --weights yolov5s.pt --source 0              # webcam
                                                             img.jpg        # image
                                                             vid.mp4        # video
                                                             path/          # directory
                                                             path/*.jpg     # glob
                                                             'https://youtu.be/Zgi9g1ksQHc'  # YouTube
                                                             'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream
Usage - formats:
    $ python path/to/detect.py --weights yolov5s.pt                 # PyTorch
                                         yolov5s.torchscript        # TorchScript
                                         yolov5s.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                                         yolov5s.xml                # OpenVINO
                                         yolov5s.engine             # TensorRT
                                         yolov5s.mlmodel            # CoreML (MacOS-only)
                                         yolov5s_saved_model        # TensorFlow SavedModel
                                         yolov5s.pb                 # TensorFlow GraphDef
                                         yolov5s.tflite             # TensorFlow Lite
                                         yolov5s_edgetpu.tflite     # TensorFlow Edge TPU
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np

import cv2
import torch
import torch.backends.cudnn as cudnn

FILE = Path(__file__).resolve() # 当前detect.py文件的绝对路径 D:\13Project_and_Dataset_Download\project\yolov5-6.1for_regional_intrusion_detection\detect.py
ROOT = FILE.parents[0]  # 获取当前文件的父目录 D:\13Project_and_Dataset_Download\project\yolov5-6.1for_regional_intrusion_detection
if str(ROOT) not in sys.path: # 判断YoloV5这个路径是否存在于模块的查询路径列表，如果不在的话就加入这个路径 否则下面导包会报错 例如：from models.common import DetectMultiBackend
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # 将ROOT转为相对路径

from models.common import DetectMultiBackend
from utils.datasets import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, time_sync





# run函数
@torch.no_grad()
def run(weights=ROOT / 'yolov5s.pt',  # model.pt path(s)
        source=ROOT / 'data/images',  # file/dir/URL/glob, 0 for webcam
        data=ROOT / 'data/coco128.yaml',  # dataset.yaml path
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=True,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        ):

    # 识别区域坐标
    # test 1 mask for certain region
    # 1,2,3,4 分别对应左上，左下，右下，右上四个点
    # hl1 = 4.5 / 10  # 监测区域高度距离图片顶部比例
    # wl1 = 5 / 10  # 监测区域高度距离图片左部比例
    # hl4 = 4.5 / 10  # 监测区域高度距离图片顶部比例
    # wl4 = 8 / 10  # 监测区域高度距离图片左部比例
    # hl2 = 9 / 10  # 监测区域高度距离图片顶部比例
    # wl2 = 5 / 10  # 监测区域高度距离图片左部比例
    # hl3 = 9 / 10  # 监测区域高度距离图片顶部比例
    # wl3 = 8 / 10  # 监测区域高度距离图片左部比例
    # # test 2 mask for certain region
    # #1,2,3,4 分别对应左上，左下，右下，右上四个点
    # hl1 = 1 / 10  # 监测区域高度距离图片顶部比例
    # wl1 = 5 / 10  # 监测区域高度距离图片左部比例
    # hl4 = 1 / 10  # 监测区域高度距离图片顶部比例
    # wl4 = 6 / 10  # 监测区域高度距离图片左部比例
    # hl2 = 9 / 10  # 监测区域高度距离图片顶部比例
    # wl2 = 5 / 10  # 监测区域高度距离图片左部比例
    # hl3 = 9 / 10  # 监测区域高度距离图片顶部比例
    # wl3 = 6 / 10  # 监测区域高度距离图片左部比例
    # # 037 左侧 mask for certain region
    # #1,2,3,4 分别对应左上，左下，右下，右上四个点
    # hl1 = 0.5 / 10    # 监测区域高度距离图片顶部比例
    # wl1 = 1.5 / 10  # 监测区域高度距离图片左部比例
    # hl4 = 0.5 / 10    # 监测区域高度距离图片顶部比例
    # wl4 = 7.5 / 10  # 监测区域高度距离图片左部比例
    # hl2 = 6.5 / 10  # 监测区域高度距离图片顶部比例
    # wl2 = 1.5 / 10  # 监测区域高度距离图片左部比例
    # hl3 = 6.5 / 10  # 监测区域高度距离图片顶部比例
    # wl3 = 7.5 / 10  # 监测区域高度距离图片左部比例
    # # 037 右侧 mask for certain region
    # #1,2,3,4 分别对应左上，左下，右下，右上四个点
    # hl1 = 0.1 / 10  # 监测区域高度距离图片顶部比例
    # wl1 = 5 / 10  # 监测区域高度距离图片左部比例
    # hl4 = 0.1 / 10  # 监测区域高度距离图片顶部比例
    # wl4 = 9.9 / 10  # 监测区域高度距离图片左部比例
    # hl2 = 5 / 10  # 监测区域高度距离图片顶部比例
    # wl2 = 5 / 10  # 监测区域高度距离图片左部比例
    # hl3 = 5 / 10  # 监测区域高度距离图片顶部比例
    # wl3 = 9.9 / 10  # 监测区域高度距离图片左部比例
    # 045
    # hl1 = 5 / 10  # 监测区域高度距离图片顶部比例
    # wl1 = 5 / 10  # 监测区域高度距离图片左部比例
    # hl4 = 5 / 10  # 监测区域高度距离图片顶部比例
    # wl4 = 9.9 / 10  # 监测区域高度距离图片左部比例
    # hl2 = 6.5 / 10  # 监测区域高度距离图片顶部比例
    # wl2 = 5 / 10  # 监测区域高度距离图片左部比例
    # hl3 = 6.5 / 10  # 监测区域高度距离图片顶部比例
    # wl3 = 9.9 / 10  # 监测区域高度距离图片左部比例



    #处理预测路径
    source = str(source) #
    #     parser.add_argument('--source', type=str, default=ROOT / 'data/video/test.mp4', help='file/dir/URL/glob, 0 for webcam')
    # save inference images【 nosave默认为False ， not nosave 则为True】
    # source.endswith('.txt')判断传入的source是否以.txt结尾 如果是source是如上代码所示 则not source.endswith('.txt')也为True
    save_img = not nosave and not source.endswith('.txt') # 由上分析的的话 save_img为True
    # IMG_FORMATS = ['bmp', 'dng', 'jpeg', 'jpg', 'mpo', 'png', 'tif', 'tiff', 'webp']  # include image suffixes
    # VID_FORMATS = ['asf', 'avi', 'gif', 'm4v', 'mkv', 'mov', 'mp4', 'mpeg', 'mpg', 'wmv']  # include video suffixes
    # 判断传入的source是否是一个文件 suffix[1:]代表后缀的切片
    # 此处的示例为.mp4 所以切片后为mp4 判断mp4是否在IMG_FORMATS + VID_FORMATS这两个变量中 此处为True
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    # source.lower()转换为小写 后判断是否以'rtsp://', 'rtmp://', 'http://', 'https://'网络流地址开头 此处为False
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    # source.isnumeric()判断是否是一个数值 可以用于判断是不是传入电脑的第n个摄像头
    # source.endswith('.txt') 略
    # is_url and not is_file 判断是不是一个网络流地址或者是不是文件
    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
    # 如果是网络流 并且是一个文件 就下载这个文件
    if is_url and is_file:
        source = check_file(source)  # download

    # 新建Directories 用于保存结果
    # project=ROOT / 'runs/detect',  # save results to project/name
    #  name='exp',  # save results to project/name
    # 把以上两个路径拼接起来作为增量路径increment_path，增量路径可以检测文件夹下exp的数字到几了 会依次累加
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    # save_txt 默认为False 如果设置为True，默认则在增量路径下再创建一个labels文件夹
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device) #选择gpu或者cpu
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data) # 判断用于加载模型的框架并加载模型
    stride, names, pt, jit, onnx, engine = model.stride, model.names, model.pt, model.jit, model.onnx, model.engine #读取模型的属性
    imgsz = check_img_size(imgsz, s=stride)  # check image size 是否为stride的倍数 如果是的话就默认保持该尺寸 否则修改为stride的倍数 stride一般为32 这里的imgsz是为了给LoadImages()赋予参数生成dataset [640,640]




    # Half
    half &= (pt or jit or onnx or engine) and device.type != 'cpu'  # FP16 supported on limited backends with CUDA
    if pt or jit:
        model.model.half() if half else model.model.float()

    # Dataloader 加载待预测的图片
    if webcam:#
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt)
        bs = len(dataset)  # batch_size

    else:
        '''
        class MyDatasetLeftUpper:
            def __init__
                self.dataset_for_scale_coords = None
                self.left=None
                self.upper=None
        '''
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)#LoadImages负责加载图片

        bs = 1  # batch_size 每次输入一张图片
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Run inference 推理产生预测结果
    model.warmup(imgsz=(1 if pt else bs, 3, *imgsz), half=half)  # 热身 初始化了一张空白图片 让模型执行了一下 再传入我们的图片
    dt, seen = [0.0, 0.0, 0.0], 0 # dt 存储时间信息 分为3个部分的时间信息 seen 计数

    for path, im, im0s, vid_cap, s in dataset:
        print('im.shape:',im.shape)
        print('imOs.shape:',im.shape)

        # 遍历图片
        # path是指路径，im是指resize后的图片，im0s是指原始图片，vid_cap=None，s是代表打印的信息
        #先执行
        # def __iter__(self):
        #     self.count = 0 # 计数
        #     return self
        '''---------------------------------开始-----------------------------------



        if webcam:
            for b in range(0,im.shape[0]):
                mask = np.zeros([im[b].shape[1], im[b].shape[2]], dtype=np.uint8)
                pts = np.array([[int(im[b].shape[2] * dataset.wl1), int(im[b].shape[1] * dataset.hl1)],  # pts1
                                [int(im[b].shape[2] * dataset.wl2), int(im[b].shape[1] * dataset.hl2)],  # pts2
                                [int(im[b].shape[2] * dataset.wl3), int(im[b].shape[1] * dataset.hl3)],  # pts3
                                [int(im[b].shape[2] * dataset.wl4), int(im[b].shape[1] * dataset.hl4)]], np.int32)
                mask = cv2.fillPoly(mask,[pts],(255,255,255))
                imgc = im[b].transpose((1, 2, 0))
                imgc = cv2.add(imgc, np.zeros(np.shape(imgc), dtype=np.uint8), mask=mask)
                im[b] = imgc.transpose((2, 0, 1))

        else:
            mask = np.zeros([im.shape[1], im.shape[2]], dtype=np.uint8)
            pts = np.array([[int(im.shape[2] * dataset.wl1), int(im.shape[1] * dataset.hl1)],  # pts1
                            [int(im.shape[2] * dataset.wl2), int(im.shape[1] * dataset.hl2)],  # pts2
                            [int(im.shape[2] * dataset.wl3), int(im.shape[1] * dataset.hl3)],  # pts3
                            [int(im.shape[2] * dataset.wl4), int(im.shape[1] * dataset.hl4)]], np.int32)# 按1、2、3、4的顺序给出坐标序列array
            mask = cv2.fillPoly(mask, [pts], (255,255,255))#填充任意形状的图型，用来绘制多边形

            # fillPoly(img, ppt, npt, 1, Scalar(255, 255, 255), lineType);

            im = im.transpose((1, 2, 0))
            im = cv2.add(im, np.zeros(np.shape(im), dtype=np.uint8), mask=mask)

            # # # 计算像素值，用于抠图
            # left = int(im.shape[1] * wl1)  # 区块左上角位置的像素点离图片左边界的距离
            # upper = int(im.shape[1] * hl1)  # 区块左上角位置的像素点离图片上边界的距离
            # right = int(im.shape[1] * wl3)  # 区块右下角位置的像素点离图片左边界的距离
            # lower = int(im.shape[1] * hl3)  # 区块右下角位置的像素点离图片上边界的距离
            # im = im[upper:lower, left:right]  # 抠出来之后的图


            

            # new_width = im.shape[1]  * 3
            # new_height =im.shape[0]  * 3
            # im = im.resize((new_width, new_height), Image.ANTIALIAS)
            # im.shape[1] = im.shape[1]  * 3
            # im.shape[0] =im.shape[0]  * 3  # TypeError: 'tuple' object does not support item assignment
            

            # 直接替换  if __name__ == '__main__':    change_size("F:\桌面\\test")


            im = im.transpose((2, 0, 1))



        --------------------------------结束--------------------------------------'''
        # 图片预处理
        t1 = time_sync() # 记录耗时
        im = torch.from_numpy(im).to(device) # CHW [3,height 640 ,width 480 ] torch.from_numpy(im)把numpy转为pytorch支持的格式 放入设备中
        im = im.half() if half else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0 归一化
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim 缺少batch这个维度，所以将它扩充一个维度
        t2 = time_sync()
        dt[0] += t2 - t1 # 第一部分的耗时

        # Inference 做预测
        visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False #visualize默认False，True的话或保存中间的特征图
        #pred的torch.Size([1,18900,85]) 18900个框
        #模型预测出来的所有检测框 augment用于判断是否要做数据增强 提高效果降低模型速度 85表示 4个坐标信息 1个置信度信息 80个类别概率
        pred = model(im, augment=augment, visualize=visualize)
        t3 = time_sync()
        dt[1] += t3 - t2 # 第二部分的耗时

        # NMS非极大值抑制
        # conf_thres: 置信度阈值
        # iou_thres: iou阈值
        # classes: 是否只保留特定的类别 默认为None
        # agnostic_nms: 进行nms是否也去除不同类别之间的框
        # max_det: 每张图片的最大目标个数 默认1000
        # pred的torch.Size([1,5,6]) 1个batch 5个框 6代表检测框信息（前四个值，左上和右下的x，y值，置信度信息，目标的类别）
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        dt[2] += time_sync() - t3 # 第三部分的耗时

        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

        # Process predictions
        # i是指每个batch的信息，det:表示5个检测框的信息，每个检测框有6个信心
        for i, det in enumerate(pred):  # per image
            seen += 1#seen是一个计数的功能
            if webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '
                '''----------------------------------开始画框----------------------------------
                cv2.putText(im0, "Detection_Region", (int(im0.shape[1] * dataset.wl1 - 5), int(im0.shape[0] * dataset.hl1 - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1.0, (255, 255, 0), 2, cv2.LINE_AA)

                pts = np.array([[int(im0.shape[1] * dataset.wl1), int(im0.shape[0] * dataset.hl1)],  # pts1
                                [int(im0.shape[1] * dataset.wl2), int(im0.shape[0] * dataset.hl2)],  # pts2
                                [int(im0.shape[1] * dataset.wl3), int(im0.shape[0] * dataset.hl3)],  # pts3
                                [int(im0.shape[1] * dataset.wl4), int(im0.shape[0] * dataset.hl4)]], np.int32)  # pts4
                # pts = pts.reshape((-1, 1, 2))
                zeros = np.zeros((im0.shape), dtype=np.uint8)
                mask = cv2.fillPoly(zeros, [pts], color=(0, 165, 255))
                im0 = cv2.addWeighted(im0, 1, mask, 0.2, 0)
                cv2.polylines(im0, [pts], True, (255, 255, 0), 3)
                # plot_one_box(dr, im0, label='Detection_Region', color=(0, 255, 0), line_thickness=2)
                ----------------------------------结束-----------------------------------'''

            else:
                # p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)
                # im0s是视频中抠出来的部分
                p, im0, frame = path, dataset.input_video_frame.copy(), getattr(dataset, 'frame', 0)
                # frame：此次取的是第几张图片
                '''----------------------------------开始--------------------------------
                cv2.putText(im0, "Detection_Region", (int(im0.shape[1] * dataset.wl1 - 5), int(im0.shape[0] * dataset.hl1 - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1.0, (255, 255, 0), 2, cv2.LINE_AA)
                pts = np.array([[int(im0.shape[1] * dataset.wl1), int(im0.shape[0] * dataset.hl1)],  # pts1
                                [int(im0.shape[1] * dataset.wl2), int(im0.shape[0] * dataset.hl2)],  # pts2
                                [int(im0.shape[1] * dataset.wl3), int(im0.shape[0] * dataset.hl3)],  # pts3
                                [int(im0.shape[1] * dataset.wl4), int(im0.shape[0] * dataset.hl4)]], np.int32)  # pts4
                # pts = pts.reshape((-1, 1, 2))
                zeros = np.zeros((im0.shape), dtype=np.uint8)
                mask = cv2.fillPoly(zeros, [pts], color=(0, 165, 255))
                im0 = cv2.addWeighted(im0, 1, mask, 0.2, 0)

                cv2.polylines(im0, [pts], True, (255, 255, 0), 3)
                ---------------------------------结束------------------------------------'''

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # im.jpg 推理结果图片保存的路径
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
            s += '%gx%g ' % im.shape[2:]  # print string #输出信息  图片shape (w, h)
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh 获得原图的宽和高的大小
            imc = im0.copy() if save_crop else im0  # for save_crop 是否要将检测的物体进行裁剪保存
            annotator = Annotator(im0, line_width=line_thickness, example=str(names)) #  # 得到一个绘图的类，类中预先存储了原图、线条宽度、类名
            if len(det):# det是框 len（det）可以判断有没有框
                # Rescale boxes from img_size to im0 size
                # 传入模型HW 1920 1088  原图HW 941 530 所以预测出来的框不能直接放到原图中 需要做一个坐标映射
                print('im.shape[2:]',im.shape[2:])
                print('det[:, :4]',det[:, :4])
                print('im0.shape',im0.shape)
                # det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round() #坐标映射功能 round()是内置函数 在utils\augmentations.py中也用到了
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round() #round()是内置函数
                # 将标注的bounding_box大小调整为和原图(cut_video)一致（因为训练时原图经过了放缩）
                print('det[:, :4]',det[:, :4])
                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class 统计每个框的类别
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string 用于打印

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file 默认不执行
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or save_crop or view_img:  # Add bbox to image
                        c = int(cls)  # integer class 获得类别
                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')# hide_labels控制隐藏标签 hide_conf控制置信度
                        annotator.box_label(xyxy, label, color=colors(c, True)) #box_label函数可以点进去看
                        if save_crop: # 默认不保存裁剪的目标图
                            save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)

            # Stream results
            im0 = annotator.result()#返回画好的图片
            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if vid_path[i] != save_path:  # new video
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                            # fps = dataset.cut_video_fps
                            # w = dataset.cut_video_w
                            # h = dataset.cut_video_h

                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                        save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer[i].write(im0)

        # Print time (inference-only)
        LOGGER.info(f'{s}Done. ({t3 - t2:.3f}s)')

    # Print results 打印输出信息
    t = tuple(x / seen * 1E3 for x in dt)  # speeds per image 平均耗时
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(weights)  # update model (to fix SourceChangeWarning)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolov5s.pt', help='model path(s)')
    '''
    可以用于指定一个训练好的模型路径，用这个模型初始化模型中一些参数（首先需要提前下载或运行程序时会自动下载）

    default
    默认为空，意义是用程序的参数权重进行初始化，而不用已经训练好的模型进行初始化。

    default
    值可设置为：

    Yolov5s.pt
    Yolov5m.pt
    Yolov5l.pt
    Yolov5x.pt
    '''

    parser.add_argument('--source', type=str, default=ROOT / 'data/video', help='file/dir/URL/glob, 0 for webcam')
    # parser.add_argument('--source', type=str, default=0, help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='(optional) dataset.yaml path')
    # parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[5120], help='inference size h,w')
    # parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[2560], help='inference size h,w')
    # parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[1280], help='inference size h,w')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[1920], help='inference size h,w')
    # parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[512], help='inference size h,w')
    # parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[320], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.4, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    # parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--line-thickness', default=2, type=int, help='bounding box thickness (pixels)')
    # parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-labels', default=True, action='store_true', help='hide labels')
    # parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--hide-conf', default=True, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # imgsz 默认为[640] 如果列表长度为1 把尺寸扩充为[640,640]
    print_args(FILE.stem, opt) # 打印所有参数信息 opt存储参数信息
    return opt

def main(opt):

    check_requirements(exclude=('tensorboard', 'thop')) # 检测依赖包是否符合要求
    run(**vars(opt)) # 执行run函数

#导包完成之后执行以下代码
if __name__ == "__main__":
    opt = parse_opt() # parse_opt是解析参数的函数 可以ctrl+左键查看 用于解析命令行传入的参数
    main(opt) # 执行main函数 并传入opt参数
