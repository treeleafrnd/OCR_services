import cv2
import numpy as np
import os


def get_template_features(template):
    sift = cv2.xfeatures2d.SIFT_create()
    kp_template, desc_template = sift.detectAndCompute(template, None)
    index_params = dict(algorithm=0, trees=5)
    search_params = dict()
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    return flann, sift, desc_template, kp_template


def get_image_features(img, sift, desc_template, flann):
    kp_img, desc_img = sift.detectAndCompute(img, None)
    matches = flann.knnMatch(desc_template, desc_img, k=2)
    return matches, kp_img


def get_perspective(matches, kp_template, kp_image, image, template):
    good_points = []
    for m, n in matches:
        if m.distance < 0.6 * n.distance:
            good_points.append(m)
    source_pts = np.float32([kp_template[m.queryIdx].pt for m in good_points]).reshape(-1, 1, 2)
    dest_pts = np.float32([kp_image[m.trainIdx].pt for m in good_points]).reshape(-1, 1, 2)
    matrix, mask = cv2.findHomography(dest_pts, source_pts, cv2.RANSAC, 5.0)
    warped_image = cv2.warpPerspective(image, matrix, (template.shape[1], template.shape[0]))
    return warped_image


def get_template(path):
    template = cv2.imread(path, 0)
    return template


def get_image(path):
    images = []
    print("Importing Images...")
    for items in os.listdir(path):
        newPath = path + '/' + items
        curImg = cv2.imread(newPath, 0)
        images.append(curImg)
    print(f"{len(images)} Images imported.")
    return images


if __name__ == '__main__':
    images = []
    template = get_template('Resources/Homo_template.jpg')
    flann, sift, desc_template, kp_template = get_template_features(template)
    images = get_image('Resources')
    for i in images:
        matches, kp_img = get_image_features(i, sift, desc_template, flann)
        warped_image = get_perspective(matches, kp_template, kp_img, i, template)
        cv2.namedWindow("Original Image", cv2.WINDOW_NORMAL)
        cv2.namedWindow("Warped Image", cv2.WINDOW_NORMAL)
        cv2.imshow("Original Image", i)
        cv2.imshow("Warped Image", warped_image)
        cv2.waitKey(0)
