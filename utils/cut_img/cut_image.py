import cv2
import matplotlib.pyplot as plt

"""
    使用OpenCV截取图片
"""


def show_cut(path, left, upper, right, lower):
    """
        原图与所截区域相比较
    :param path: 图片路径
    :param left: 区块左上角位置的像素点离图片左边界的距离
    :param upper：区块左上角位置的像素点离图片上边界的距离
    :param right：区块右下角位置的像素点离图片左边界的距离
    :param lower：区块右下角位置的像素点离图片上边界的距离
     故需满足：lower > upper、right > left
    """

    img = cv2.imread(path)


    print("This image's size: {}".format(img.shape))  # (H, W, C)

    plt.figure("Image Contrast")
    plt.subplot(1, 2, 1)
    plt.title('origin')
    plt.imshow(img)  # 展示图片的颜色会改变
    plt.axis('off')

    cropped = img[upper:lower, left:right]

    plt.subplot(1, 2, 2)
    plt.title('roi')
    plt.imshow(cropped)
    plt.axis('off')
    plt.show()


def image_cut_save(path, left, upper, right, lower, save_path):
    """
        所截区域图片保存
    :param path: 图片路径
    :param left: 区块左上角位置的像素点离图片左边界的距离
    :param upper：区块左上角位置的像素点离图片上边界的距离
    :param right：区块右下角位置的像素点离图片左边界的距离
    :param lower：区块右下角位置的像素点离图片上边界的距离
     故需满足：lower > upper、right > left
    :param save_path: 所截图片保存位置
    """
    img = cv2.imread(path)  # 打开图像
    cropped = img[upper:lower, left:right]

    # 保存截取的图片
    cv2.imwrite(save_path, cropped)


if __name__ == '__main__':
    pic_path = r'D:\13Project_and_Dataset_Download\project\yolov5-6.1for_regional_intrusion_detection2\data\images\IMG_350.jpg'
    pic_save_dir_path = 'cut.jpg'
    left, upper, right, lower = 1, 145, 165, 170
    show_cut(pic_path, left, upper, right, lower)
    image_cut_save(pic_path, left, upper, right, lower, pic_save_dir_path)

