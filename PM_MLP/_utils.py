import cv2
import math
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import torch
import torch.nn as nn
import torch.nn.functional as F

# %%
class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def Saturation_Map(im):
    saturation = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    return saturation[:, :, 1]


def DarkChannel(im, sz):
    '''
    :param im: Input image
    :param sz: Size Window
    :return: THe image after Dark Channel
    '''
    b, g, r = cv2.split(im)
    dc = cv2.min(cv2.min(r, g), b)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (sz, sz))
    dark = cv2.erode(dc, kernel)
    return dark


def AtmLight(im, dark):
    [h, w] = im.shape[:2]
    imsz = h * w
    numpx = int(max(math.floor(imsz / 1000), 1))
    darkvec = dark.reshape(imsz)
    imvec = im.reshape(imsz, 3)

    indices = darkvec.argsort()
    indices = indices[imsz - numpx::]

    atmsum = np.zeros([1, 3])
    for ind in range(1, numpx):
        atmsum = atmsum + imvec[indices[ind]]

    A = atmsum / numpx
    return A


def TransmissionEstimate(im, A, sz):
    omega = 0.95
    im3 = np.empty(im.shape, im.dtype)

    for ind in range(0, 3):
        im3[:, :, ind] = im[:, :, ind] / A[0, ind]

    transmission = 1 - omega * DarkChannel(im3, sz)
    return transmission


def Guidedfilter(im, p, r, eps):
    mean_I = cv2.boxFilter(im, cv2.CV_64F, (r, r))
    mean_p = cv2.boxFilter(p, cv2.CV_64F, (r, r))
    mean_Ip = cv2.boxFilter(im * p, cv2.CV_64F, (r, r))
    cov_Ip = mean_Ip - mean_I * mean_p

    mean_II = cv2.boxFilter(im * im, cv2.CV_64F, (r, r))
    var_I = mean_II - mean_I * mean_I

    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I

    mean_a = cv2.boxFilter(a, cv2.CV_64F, (r, r))
    mean_b = cv2.boxFilter(b, cv2.CV_64F, (r, r))

    q = mean_a * im + mean_b
    return q


def TransmissionRefine(im, et):
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    gray = np.float64(gray) / 255
    r = 60
    eps = 0.0001
    t = Guidedfilter(gray, et, r, eps)

    return t


def Recover(im, t, A, tx=0.1):
    res = np.empty(im.shape, im.dtype)
    t = cv2.max(t, tx)

    for ind in range(0, 3):
        res[:, :, ind] = (im[:, :, ind] - A[0, ind]) / t + A[0, ind]

    return res

def get_rmse(label, pred):
    return mean_squared_error(label, pred, squared=False)

def get_mae(label, pred):
    return mean_absolute_error(label, pred)

def get_r2score(label, pred):
    return r2_score(label, pred)

def get_nmge(label, pred):
    res = torch.sum(torch.abs(pred - label))
    nmge = res / torch.sum(torch.abs(label))
    return nmge

def get_metrics(label, pred):
    rmse = get_rmse(label, pred)
    mae = get_mae(label, pred)
    r2_score = get_r2score(label, pred)
    nmge = get_nmge(label, pred)
    return rmse, mae, r2_score, nmge


if __name__ == '__main__':
    import sys

    try:
        fn = sys.argv[1]
    except:
        fn = './image/15.png'


    def nothing(*argv):
        pass


    src = cv2.imread(fn)

    I = src.astype('float64') / 255

    dark = DarkChannel(I, 15)
    A = AtmLight(I, dark)
    te = TransmissionEstimate(I, A, 15)
    t = TransmissionRefine(src, te)
    J = Recover(I, t, A, 0.1)

    cv2.imshow("dark", dark)
    cv2.imshow("t", t)
    cv2.imshow("te", te)
    cv2.imshow('I', src)
    cv2.imshow('J', J)
    cv2.imwrite("./image/J.png", J * 255)
    cv2.waitKey()