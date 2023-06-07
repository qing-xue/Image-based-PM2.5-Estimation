import PIL
import numpy as np
from PIL import Image
import cv2


def DarkChannel(im, win=15):
    """ 求暗通道图
    :param im: 输入 3 通道图
    :param win: 图像腐蚀窗口, [win x win]
    :return: 暗通道图
    """
    if isinstance(im, PIL.Image.Image):
        im = cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR)

    b, g, r = cv2.split(im)
    dc = cv2.min(cv2.min(r, g), b)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (win, win))  # 结构元素
    dark = cv2.erode(dc, kernel)                                    # 腐蚀操作
    return Image.fromarray(dark, mode='L')


def SaturationMap(im):
    im = Image.fromarray(np.uint8(im))
    img_hsv = im.convert("HSV")
    h, s, v = img_hsv.split()  # 分离三通道
    return s
