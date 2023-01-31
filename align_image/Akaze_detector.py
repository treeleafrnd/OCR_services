import cv2
import numpy as np


def get_template_features(template):
    akaze = cv2.AKAZE_create()
    kp_template, desc_template = akaze.detectAndCompute(template, None)
    index_params = dict(algorithm=0, trees=5)
    search_params = dict()
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    return flann, akaze, desc_template, kp_template


def get_image_features(img, akaze, desc_template, flann):
    kp_img, desc_img = akaze.detectAndCompute(img, None)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(desc_template, desc_img)
    matches = sorted(matches, key=lambda x: x.distance)
    return matches, kp_img


def get_aligned_img(matches, kp_template, kp_image, image, template):
    source_pts = np.float32([kp_template[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dest_pts = np.float32([kp_image[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    img_keyed = cv2.drawKeypoints(image, kp_image, template)
    matrix, mask = cv2.findHomography(dest_pts, source_pts, cv2.RANSAC, 5.0)
    warped_image = cv2.warpPerspective(image, matrix, (template.shape[1], template.shape[0]))
    return warped_image, img_keyed
