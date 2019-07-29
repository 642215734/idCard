import cv2
import numpy as np

'''
倾斜矫正
投影分割，行->列->行

'''
def segment(path='./5.png', root='./', number=0, dsize=100):
    # dsize归一化处理的图像大小
    stop=number+3
    img = cv2.imread(path)
    data = np.array(img)
    len_x = data.shape[0]
    len_y = data.shape[1]
    min_val = 10  # 设置最小的文字像素高度，防止切分噪音字符

    start_i = -1
    end_i = -1
    rowPairs = []  # 存放每行的起止坐标

    # 行分割
    for i in range(len_x):
        if (not data[i].all() and start_i < 0):
            start_i = i
        elif (not data[i].all()):
            end_i = i
        elif (data[i].all() and start_i >= 0):
            # print(end_i - start_i)
            if (end_i - start_i >= min_val):
                rowPairs.append((start_i, end_i))
            start_i, end_i = -1, -1

    print(rowPairs)

    # 列分割
    start_j = -1
    end_j = -1
    min_val_word = 5  # 最小文字像素长度
    # 分割后保存编号
    for start, end in rowPairs:
        for j in range(len_y):
            if (not data[start: end, j].all() and start_j < 0):
                start_j = j
            elif (not data[start: end, j].all()):
                end_j = j
            elif (data[start: end, j].all() and start_j >= 0):
                if (end_j - start_j >= min_val_word):
                    # print(end_j - start_j)
                    tmp = data[start:end, start_j: end_j]
                    # cv2.imshow('demo', tmp)
                    # cv2.waitKey(0)
                    # print(tmp.shape)

                    start_i = -1
                    end_i = -1
                    for i in range(tmp.shape[0]):
                        if (not tmp[i].all() and start_i < 0):
                            start_i = i
                        elif (not tmp[i].all()):
                            end_i = i
                        elif (tmp[i].all() and start_i >= 0):
                            # print(end_i - start_i)
                            if (end_i - start_i >= min_val):
                                pass

                    tmp = tmp[start_i:end_i, :]
                    cv2.imshow('demo', tmp)
                    cv2.waitKey(0)

                    im2save = cv2.resize(tmp, (dsize, dsize))  # 归一化处理
                    cv2.imwrite(root + '/%d.png' % number, im2save)
                    number += 1
                    print("%d  pic" % number)
                    # 只要前三个字
                    if number == stop:
                        break
                start_j, end_j = -1, -1


if __name__ == '__main__':
    segment('./7.png', './data/train',17)
