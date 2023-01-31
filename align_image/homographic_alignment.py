import cv2
import numpy as np


def get_template_features(template_img):
    sift = cv2.xfeatures2d.SIFT_create()
    kp_template, desc_template = sift.detectAndCompute(template_img, None)
    index_params = dict(algorithm=0, trees=5)
    search_params = dict()
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    return flann, sift, desc_template, kp_template


def get_image_features(img, sift, desc_template, flann):
    kp_img, desc_img = sift.detectAndCompute(img, None)
    matches = flann.knnMatch(desc_template, desc_img, k=2)
    return matches, kp_img


def get_aligned_img(matches, kp_template, kp_image, image, template):
    """
    Main function to return aligned image
    """
    good_points = []
    for m, n in matches:
        if m.distance < 0.6 * n.distance:
            good_points.append(m)
    source_pts = np.float32([kp_template[m.queryIdx].pt for m in good_points]).reshape(-1, 1, 2)
    dest_pts = np.float32([kp_image[m.trainIdx].pt for m in good_points]).reshape(-1, 1, 2)
    matrix, mask = cv2.findHomography(dest_pts, source_pts, cv2.RANSAC, 5.0)
    warped_image = cv2.warpPerspective(image, matrix, (template.shape[1], template.shape[0]))
    return warped_image
