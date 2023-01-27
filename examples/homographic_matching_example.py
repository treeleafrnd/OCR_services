"""
-- Created by Pravesh Budhathoki
-- Treeleaf Technologies Pvt. Ltd.
-- Created on 2023-01-27
"""
import cv2

from align_image.homographic_alignment import get_template_features, get_image_features, \
    get_aligned_img
from utils.imutil import read_image, get_image

if __name__ == '__main__':
    # images = []
    template = read_image('test_images/Homo_template.jpg')
    flann, sift, desc_template, kp_template = get_template_features(template)
    images = get_image('test_images')
    for i in images:
        matches, kp_img = get_image_features(i, sift, desc_template, flann)
        warped_image = get_aligned_img(matches, kp_template, kp_img, i, template)
        cv2.namedWindow("Original Image", cv2.WINDOW_NORMAL)
        cv2.namedWindow("Warped Image", cv2.WINDOW_NORMAL)
        cv2.imshow("Original Image", i)
        cv2.imshow("Warped Image", warped_image)
        cv2.waitKey(0)
