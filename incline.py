import os
import cv2
import math
import random
import numpy as np
from scipy import misc, ndimage
import matplotlib.pyplot as plt


# 旋转angle角度，缺失背景白色（255, 255, 255）填充
def rotate_bound_white_bg(image, angle):
    # grab the dimensions of the image and then determine the
    # center
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    # -angle位置参数为角度参数负值表示顺时针旋转; 1.0位置参数scale是调整尺寸比例（图像缩放参数），建议0.75
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    # perform the actual rotation and return the image
    # borderValue 缺失背景填充色彩，此处为白色，可自定义
    return cv2.warpAffine(image, M, (nW, nH), borderValue=(255, 255, 255))
    # borderValue 缺省，默认是黑色（0, 0 , 0）
    # return cv2.warpAffine(image, M, (nW, nH))


filepath = './'

img = cv2.imread('./7.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 50, 150, apertureSize=3)

# 霍夫变换
lines = cv2.HoughLines(edges, 1, np.pi / 180, 0)
rotate_angle = 0
for rho, theta in lines[0]:
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a * rho
    y0 = b * rho
    x1 = int(x0 + 1000 * (-b))
    y1 = int(y0 + 1000 * (a))
    x2 = int(x0 - 1000 * (-b))
    y2 = int(y0 - 1000 * (a))
    if x1 == x2 or y1 == y2:
        continue
    t = float(y2 - y1) / (x2 - x1)
    rotate_angle = math.degrees(math.atan(t))
    if rotate_angle > 45:
        rotate_angle = -90 + rotate_angle
    elif rotate_angle < -45:
        rotate_angle = 90 + rotate_angle
print("rotate_angle : " + str(rotate_angle))
rotate_img = rotate_bound_white_bg(img, -rotate_angle)
# rotate_img = ndimage.rotate(img, rotate_angle)
misc.imsave('./in.png', rotate_img)
cv2.imshow("img", rotate_img)
cv2.waitKey(0)



