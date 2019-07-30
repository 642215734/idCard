import cv2
import math
import numpy as np
import imageio

import numpy as np
import os
import cv2
import math

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

def rotate(image, angle, center=None, scale=1.0):
    (w, h) = image.shape[0:2]
    if center is None:
        center = (w // 2, h // 2)
    wrapMat = cv2.getRotationMatrix2D(center, angle, scale)
    return cv2.warpAffine(image, wrapMat, (h, w))


# 使用矩形框
def incline_api(filename='./data/demo_pre.png', savepath='./in.png'):
    # 读取图片，灰度化
    src = cv2.imread(filename)

    gray = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)

    # 图像取非
    grayNot = cv2.bitwise_not(gray)

    # 二值化
    threImg = cv2.threshold(grayNot, 100, 255, cv2.THRESH_BINARY, )[1]

    # 获得有文本区域的点集,求点集的最小外接矩形框，并返回旋转角度
    coords = np.column_stack(np.where(threImg > 0))
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(angle + 90)
    else:
        angle = -angle

    # 仿射变换，将原图校正
    dst = rotate_bound_white_bg(src, -angle)
    # cv2.imshow("dst", dst)
    # cv2.waitKey()
    print(angle)
    imageio.imsave(savepath, dst)


if __name__ == "__main__":
    incline_api(filename='./data/demo_pre.png', savepath='./in.png')

























#
# # 旋转angle角度，缺失背景白色（255, 255, 255）填充
# def rotate_bound_white_bg(image, angle):
#     # grab the dimensions of the image and then determine the
#     # center
#     (h, w) = image.shape[:2]
#     (cX, cY) = (w // 2, h // 2)
#
#     # grab the rotation matrix (applying the negative of the
#     # angle to rotate clockwise), then grab the sine and cosine
#     # (i.e., the rotation components of the matrix)
#     # -angle位置参数为角度参数负值表示顺时针旋转; 1.0位置参数scale是调整尺寸比例（图像缩放参数），建议0.75
#     M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
#     cos = np.abs(M[0, 0])
#     sin = np.abs(M[0, 1])
#
#     # compute the new bounding dimensions of the image
#     nW = int((h * sin) + (w * cos))
#     nH = int((h * cos) + (w * sin))
#
#     # adjust the rotation matrix to take into account translation
#     M[0, 2] += (nW / 2) - cX
#     M[1, 2] += (nH / 2) - cY
#
#     # perform the actual rotation and return the image
#     # borderValue 缺失背景填充色彩，此处为白色，可自定义
#     return cv2.warpAffine(image, M, (nW, nH), borderValue=(255, 255, 255))
#     # borderValue 缺省，默认是黑色（0, 0 , 0）
#     # return cv2.warpAffine(image, M, (nW, nH))
#
#
# def incline_api(filename='./7.png', savepath='./in.png'):
#     img = cv2.imread(filename)
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     edges = cv2.Canny(gray, 50, 150, apertureSize=3)
#
#     # 霍夫变换
#     lines = cv2.HoughLines(edges, 1, np.pi / 180, 0)
#     rotate_angle = 0
#     for rho, theta in lines[0]:
#         a = np.cos(theta)
#         b = np.sin(theta)
#         x0 = a * rho
#         y0 = b * rho
#         x1 = int(x0 + 1000 * (-b))
#         y1 = int(y0 + 1000 * (a))
#         x2 = int(x0 - 1000 * (-b))
#         y2 = int(y0 - 1000 * (a))
#         if x1 == x2 or y1 == y2:
#             continue
#         t = float(y2 - y1) / (x2 - x1)
#         rotate_angle = math.degrees(math.atan(t))
#         if rotate_angle > 45:
#             rotate_angle = -90 + rotate_angle
#         elif rotate_angle < -45:
#             rotate_angle = 90 + rotate_angle
#     print("rotate_angle : " + str(rotate_angle))
#     rotate_img = rotate_bound_white_bg(img, -rotate_angle)
#     # rotate_img = ndimage.rotate(img, rotate_angle)
#     imageio.imsave(savepath, rotate_img)
#     cv2.imshow("img", rotate_img)
#     cv2.waitKey(0)
#     return rotate_img
#
#
# if __name__ == '__main__':
#     rotate_img=incline_api(filename='./data/demo_pre.png', savepath='./in.png')
