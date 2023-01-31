import cv2
import numpy as np
import math


def mouseKey(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        param.append([x, y])
    # return param


if __name__ == '__main__':
    param = []
    image = cv2.imread('Resources/Citizenship/all/printed/high_quality/FRONT/f11.jpg')
    while True:
        cv2.namedWindow('Image', cv2.WINDOW_NORMAL)
        cv2.setMouseCallback('Image', mouseKey, param)
        print(param)
        if len(param) == 4:
            pts1 = np.float32(param)
            pts2 = np.float32([[0, 0], [500, 0], [0, 400], [500, 400]])
            M = cv2.getPerspectiveTransform(pts1, pts2)
            dst = cv2.warpPerspective(image, M, (500, 400))
            cv2.imshow('Transformed', dst)
        cv2.imshow('Image', image)
        cv2.waitKey(1)
