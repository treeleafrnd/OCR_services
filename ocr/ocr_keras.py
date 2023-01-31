import cv2
import keras_ocr
import numpy as np
import os
import tensorflow as tf


def keras_pipeline():
    return keras_ocr.pipeline.Pipeline()


def get_image(path):
    image = [keras_ocr.tools.read(ocr_img_keras) for ocr_img_keras in
             [  # 'Resources/High_res/test1.png',
                 # 'Resources/c_backside/document.png',
                 path]]
    return image


def recognize_image(img):
    prediction_groups = pipeline.recognize(img)
    return prediction_groups


def plot_result(image, prediction_groups):
    for text, bbox in prediction_groups[0]:
        cv2.polylines(image, np.int32([bbox]), 2, (0, 255, 0), 2)
    return image


def get_text(prediction_groups):
    predicted = prediction_groups[0]
    for text, bbox in predicted:
        print(text)


if __name__ == '__main__':
    pipeline = keras_pipeline()
    path = 'test_images'
    print("Images Imported and Now processing...")
    for dir in os.listdir(path):
        newPath = path + '/' + dir
        image = get_image(newPath)
        #img = noise_removal(image[0])
        #img_deskew = dskew_img(img)
        image.pop()
        image.append(img_deskew)
        prediction_groups = recognize_image(image)
        final_image = plot_result(img_deskew, prediction_groups)
        get_text(prediction_groups)
        cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
        cv2.imshow("Image", final_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
