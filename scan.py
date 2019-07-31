from transform import four_point_transform
import cv2, imutils
import imgEnhance


def preProcess(image):
    ratio = image.shape[0] / 500.0
    image = imutils.resize(image, height=500)

    grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gaussImage = cv2.GaussianBlur(grayImage, (5, 5), 0)
    edgedImage = cv2.Canny(gaussImage, 75, 200)

    cnts = cv2.findContours(edgedImage.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if imutils.is_cv2() else cnts[1]
    docCnt = None

    # 确保至少有一个轮廓被找到
    if len(cnts) > 0:
        # 将轮廓按大小降序排序
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

        # 对排序后的轮廓循环处理
        for c in cnts:
            # 获取近似的轮廓
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)

            # 如果我们的近似轮廓有四个顶点，那么就认为找到了答题卡
            if len(approx) == 4:
                docCnt = approx
                break

    return docCnt, ratio


if __name__ == "__main__":
    image = cv2.imread("./Location_anchor/2.jpg")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 75, 200)

    # find contours in the edge map, then initialize
    # the contour that corresponds to the document
    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[1] if imutils.is_cv3() else cnts[0]
    docCnt = None

    # ensure that at least one contour was found
    if len(cnts) > 0:
        # sort the contours according to their size in
        # descending order
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

        # loop over the sorted contours
        for c in cnts:
            # approximate the contour
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)

            # if our approximated contour has four points,
            # then we can assume we have found the paper
            if len(approx) == 4:
                docCnt = approx
                break

    warped = four_point_transform(image, docCnt.reshape(4, 2))

    enhancer = imgEnhance.Enhancer()
    enhancedImg = enhancer.gamma(warped, 1.63)
    image = imutils.resize(image, height=151, width=240)
    enhancedImg = imutils.resize(enhancedImg, height=151, width=240)

    # shape = enhancedImg.shape
    # h, w = shape[0], shape[1]
    # print(w, h)

    anchor = enhancedImg[32:65, 65:225]
    # anchor = enhancedImg[int(h * 0.1):int(h * 0.9), int(w * 0.2):int(w * 0.5)]
    anchor = imutils.resize(anchor, height=151, width=240)

    cv2.imshow("org", image)
    cv2.imshow("gamma", enhancedImg)
    cv2.imshow("anchor", anchor)
    cv2.waitKey(0)

    cv2.destroyAllWindows()
