'''

图像二值化&图像增广
'''

import d2lzh as d2l
import mxnet as mx
from mxnet import autograd, gluon, image, init, nd
from mxnet.gluon import data as gdata, loss as gloss, utils as gutils
import sys
import time
import numpy as np
from PIL import Image
import cv2
from matplotlib import pyplot as plt


def prepro_api(path='./data/raw_true/2.png', dest='./data/demo_pre.png'):
    # d2l.set_figsize()
    # img = image.imread(path)

    # d2l.plt.imshow(img.asnumpy())

    # def apply(img, aug, num_rows=2, num_cols=4, scale=1.5):
    #     Y = [aug(img) for _ in range(num_rows * num_cols)]
    #     d2l.show_images(Y, num_rows, num_cols, scale)

    # apply(img, gdata.vision.transforms.RandomFlipLeftRight())
    # apply(img, gdata.vision.transforms.RandomFlipTopBottom())
    # shape_aug = gdata.vision.transforms.RandomResizedCrop(
    #     (1200, 300), scale=(0.6, 1), ratio=(0.1, 0.4))
    # apply(img, shape_aug)
    # transform_test = gdata.vision.transforms.Compose([
    #     gdata.vision.transforms.ToTensor(),
    #     gdata.vision.transforms.Normalize([0.4914, 0.4822, 0.4465], [0.20230, 19940, 2010])])
    # color_aug = gdata.vision.transforms.RandomColorJitter(
    #     brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5)
    # apply(img, color_aug)
    # d2l.plt.show()
    # img = cv2.imread(path, 0)
    # img = cv2.medianBlur(img, 5)
    # ret, th1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    # th2 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, \
    #                             cv2.THRESH_BINARY, 31, 2)
    # th3 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, \
    #                             cv2.THRESH_BINARY, 31, 2)
    # titles = ['Original Image', 'Global Thresholding (v = 127)',
    #           'Adaptive Mean Thresholding', 'Adaptive Gaussian Thresholding']
    # images = [img, th1, th2, th3]
    # for i in range(4):
    #     plt.subplot(2, 2, i + 1), plt.imshow(images[i], 'gray')
    #     plt.title(titles[i])
    #     plt.xticks([]), plt.yticks([])
    # plt.show()

    img = Image.open(path)
    img = img.convert("L")

    WHITE, BLACK = 255, 0
    th=np.array(img).mean()-30

    img = img.point(lambda x: WHITE if x > th else BLACK)
    img = img.convert('1')
    img.save(dest)


if __name__ == '__main__':
    prepro_api()
