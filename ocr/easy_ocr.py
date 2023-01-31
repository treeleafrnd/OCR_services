#import TextBlob
import easyocr
import cv2
import numpy as np


def extract_text_easyocr(img):
    reader = easyocr.Reader(['en', 'ne'], gpu=False)
    text = reader.readtext(img)
    for bbox, txt, conf in text:
        cv2.polylines(img, np.int32([bbox]), 2, (0, 255, 0), 2)
    return text, img
