import cv2
import numpy as np
from skimage.transform import rotate
import deskew


def dskew_img(img):
    angle = deskew.determine_skew(img)
    img = rotate(img, angle, resize=True) * 255
    print(f'Angle of Rotation:{angle}')
    return img.astype(np.uint8)


def noise_removal(img):
    img = cv2.fastNlMeansDenoising(img, None, 10, 7, 21)
    # img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 12)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img