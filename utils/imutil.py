import os
import cv2


def read_image(path):
    img = cv2.imread(path, 0)
    return img


def get_image(path):
    images = []
    print("Importing Images...")
    for items in os.listdir(path):
        newPath = path + '/' + items
        curImg = cv2.imread(newPath, 0)
        images.append(curImg)
    print(f"{len(images)} Images imported.")
    return images
