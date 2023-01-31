import cv2
import pytesseract
from pytesseract import Output


def bbox_c_img(img, copy):
    config_type = r'-l eng+nep --oem 3 --psm 11'
    labels = pytesseract.image_to_data(img, output_type=Output.DICT, config=config_type)
    n_boxes = len(labels['text'])
    for box in range(n_boxes):
        confidence = int(labels['conf'][box])
        x, y, w, h = (labels['left'][box], labels['top'][box], labels['width'][box], labels['height'][box])
        copy = cv2.rectangle(copy, (x, y), (x + w, y + h), (255, 0, 0), 2)
    return copy


def ocr_c(img):
    config_type = r'-l eng+nep --oem 3 --psm 11'
    return pytesseract.image_to_string(img, config=config_type)
