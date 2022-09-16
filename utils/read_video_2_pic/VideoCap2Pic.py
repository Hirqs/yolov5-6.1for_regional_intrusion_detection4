import cv2


def getFrame(videoPath, svPath):
    cap = cv2.VideoCapture(videoPath)
    count = cap.get(7)  # 获取总帧数
    numFrame = 0
    while True:
        if cap.grab():
            flag, frame = cap.retrieve()
            if not flag:
                continue
            else:
                # cv2.imshow('video', frame)
                numFrame += 1
                newPath = svPath + "antitank_reactor_" + str(numFrame) + ".jpg" # 设置想保存的文件名，名字不要重复，可能会覆盖
                if numFrame % 100 == 0:  # 这里设置每多少帧保存一次图片，我这里是写的50，根据需要修改
                    cv2.imencode('.jpg', frame)[1].tofile(newPath)
                if numFrame % 1 == 0:  # 每过200帧打印一下，方便知道程序在运行
                    print("Have read " + str(numFrame) + " frames.")
        if numFrame == count:
            break
        if cv2.waitKey(10) == 27:
            break
    print(numFrame)


# 第一个参数是视频源文件，第二个参数是要保存到哪个文件夹，自行动创建，没写创建文件夹的代码，我这里是保存当前目录下面的images文件夹
getFrame(r"D:\13Project_and_Dataset_Download\project\yolov5-6.1for_regional_intrusion_detection\data\video\DJI_0037.MP4", "./test3/")

