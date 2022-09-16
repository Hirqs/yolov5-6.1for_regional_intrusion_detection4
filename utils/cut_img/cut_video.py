import cv2
#函数介绍：将已有的视频某一部分截取下来保存为视频。例如：将(height, width)为（960， 2560）
#的视频转换为(height, width)为（960， 1280）的视频（为视频图像的某一部分）。
def split_video(input_video, output_video):
    # # 037 右侧 mask for certain region
    # #1,2,3,4 分别对应左上，左下，右下，右上四个点
    hl1 = 0.1 / 10  # 监测区域高度距离图片顶部比例
    wl1 = 5 / 10  # 监测区域高度距离图片左部比例
    hl4 = 0.1 / 10  # 监测区域高度距离图片顶部比例
    wl4 = 9.9 / 10  # 监测区域高度距离图片左部比例
    hl2 = 5 / 10  # 监测区域高度距离图片顶部比例
    wl2 = 5 / 10  # 监测区域高度距离图片左部比例
    hl3 = 5 / 10  # 监测区域高度距离图片顶部比例
    wl3 = 9.9 / 10  # 监测区域高度距离图片左部比例
    video_caputre = cv2.VideoCapture(input_video)
    # get video parameters
    fps = video_caputre.get(cv2.CAP_PROP_FPS)
    width = video_caputre.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = video_caputre.get(cv2.CAP_PROP_FRAME_HEIGHT)

    # # 计算像素值，用于抠图
    left = int(width * wl1)  # 区块左上角位置的像素点离图片左边界的距离
    upper = int(height * hl1)  # 区块左上角位置的像素点离图片上边界的距离
    right = int(width * wl3)  # 区块右下角位置的像素点离图片左边界的距离
    lower = int(height * hl3)  # 区块右下角位置的像素点离图片上边界的距离
    print("fps:", fps)
    print("width:", width)
    print("height:", height)
    # 定义截取尺寸,后面定义的每帧的h和w要于此一致，否则视频无法播放
    size = (right-left, lower-upper)
    # https://blog.csdn.net/weixin_41581849/article/details/120422265
    # cv2.VideoWriter_fourcc('X', '2', '6', '4'), 该参数是较新的MPEG-4编码,产生的文件较小,文件扩展名应为.mp4
    # cv2.VideoWriter_fourcc('P', 'I', 'M', 'I'), 该参数是较旧的MPEG-1编码,文件名后缀为.avi
    # cv2.VideoWriter_fourcc('M', 'P', '4', '2'), 该参数是MPEG-2编码,产生的文件不会特别大,文件名后缀为.avi
    # cv2.VideoWriter_fourcc('D', 'I', 'V', '3'), 该参数是MPEG-3编码,产生的文件不会特别大,文件名后缀为.avi
    # cv2.VideoWriter_fourcc('D', 'I', 'V', 'X'), 该参数是MPEG-4编码,产生的文件不会特别大,文件名后缀为.avi
    # cv2.VideoWriter_fourcc('X', 'V', 'I', 'D'), 该参数是较旧的MPEG-4编码,产生的文件不会特别大,文件名后缀为.avi
    # cv2.VideoWriter_fourcc('M', 'P', '4', 'V'), 该参数也是较旧的MPEG-4编码,产生的文件不会特别大,文件扩展名应为.m4v
    # cv2.VideoWriter_fourcc('T', 'H', 'E', 'O'), 该参数是Ogg Vorbis,产生的文件相对较大,文件名后缀为.ogv
    # cv2.VideoWriter_fourcc('F', 'L', 'V', '1'), 该参数是Flash视频,产生的文件相对较大,文件名后缀为.flv
    # cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 该参数是motion-jpeg编码,产生的文件较大,文件名后缀为.avi
    # cv2.VideoWriter_fourcc('I', '4', '2', '0'),该参数是未压缩的YUV编码,4:2:0色度子采样,这种编码广泛兼容,但会产生特别大的文件,文件扩展名应为.avi ```
    fourcc = cv2.VideoWriter_fourcc('M', 'P', '4', 'V') # VideoWriter函数是opencv中用来生成视频的函数
    # 创建视频写入对象
    videp_write = cv2.VideoWriter(output_video, fourcc, fps, size)

    print('Start!!!')
    # 读取视频帧
    success, frame_src = video_caputre.read()  # (960, 2560, 3)  # (height, width, channel)
    while success and not cv2.waitKey(1) == 27:  # 读完退出或者按下 esc 退出

        frame_target = frame_src[upper:lower, left:right]  # 抠出来之后的图

        # 写入视频文件
        videp_write.write(frame_target)
        # 不断读取
        success, frame_src = video_caputre.read()

    print("Finished!!!")
    video_caputre.release()

if __name__ == '__main__':
    input_file = r'D:\13Project_and_Dataset_Download\project\yolov5-6.1for_regional_intrusion_detection2\data\video\input.mp4'
    output_file = r'D:\13Project_and_Dataset_Download\project\yolov5-6.1for_regional_intrusion_detection2\data\video\output.mp4'
    split_video(input_video=input_file, output_video=output_file)

