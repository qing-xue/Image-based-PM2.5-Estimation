import numpy as np
from cv2 import cv2
from PIL import Image

import utils


class IA_Features():

    def __init__(self):
        super(IA_Features, self).__init__()
        self.w = (1, 3, 5)  # Page 9

    def getInfo(self, img_):
        dc = utils.DarkChannel(img_, win=1)  # Note the difference between Darkness & Dark Channel
        sm = utils.SaturationMap(img_)

        info = list()
        for i in self.w:
            dc2 = self._process_img(np.array(dc), i)  # For show: Image.fromarray(x).convert('RGB')
            H1 = self._calc_array(dc2)
            sm2 = self._process_img(np.array(sm), i)
            H2 = self._calc_array(sm2)
            info.append(H1)
            info.append(H2)

        return tuple(info)


    def _calc_array(self, img):
        """Information Entropy"""
        entropy = []

        hist = cv2.calcHist([img], [0], None, [256], [0, 255])
        total_pixel = img.shape[0] * img.shape[1]

        for item in hist.flatten():
            probability = item / total_pixel
            if probability == 0:
                en = 0
            else:
                en = -1 * probability * (np.log(probability) / np.log(2))
            entropy.append(en)

        sum_en = np.sum(entropy)
        return sum_en

    def _process_img(self, img_, w):
        """Equation (4)"""
        img = img_.copy().astype(np.int16)  # cannot be uint8

        c = np.ones_like(img) * 255         # delta l = 255
        t = np.add(w * (np.subtract(img, c)), c)
        img = np.minimum(c, t)              # TODO: can be removed

        img = np.clip(img, 0, 255)
        return img.astype(np.uint8)