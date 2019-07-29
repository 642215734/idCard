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

path = './data/7.png'

d2l.set_figsize()
img = image.imread(path)

th = 120
d2l.plt.imshow(img.asnumpy())


def apply(img, aug, num_rows=2, num_cols=4, scale=1.5):
    Y = [aug(img) for _ in range(num_rows * num_cols)]
    d2l.show_images(Y, num_rows, num_cols, scale)


# apply(img, gdata.vision.transforms.RandomFlipLeftRight())
# apply(img, gdata.vision.transforms.RandomFlipTopBottom())
shape_aug = gdata.vision.transforms.RandomResizedCrop(
    (1200, 300), scale=(0.6, 1), ratio=(0.1, 0.4))
apply(img, shape_aug)
transform_test = gdata.vision.transforms.Compose([
    gdata.vision.transforms.ToTensor(),
    gdata.vision.transforms.Normalize([0.4914, 0.4822, 0.4465], [0.20230, 19940, 2010])])
# color_aug = gdata.vision.transforms.RandomColorJitter(
#     brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5)
# apply(img, color_aug)
d2l.plt.show()

img = Image.open(path)
img = img.convert("L")

WHITE, BLACK = 255, 0

img = img.point(lambda x: WHITE if x > th else BLACK)
img = img.convert('1')
img.save(path[-5:])
