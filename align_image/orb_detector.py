import cv2
import numpy as np
import os


def noise_removal(img):
    img = cv2.fastNlMeansDenoising(img, None, 10, 7, 21)
    return img


def get_template_features(template):
    orb = cv2.ORB_create(2000)
    kp_template, desc_template = orb.detectAndCompute(template, None)
    index_params = dict(algorithm=0, trees=5)
    search_params = dict()
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    return flann, orb, desc_template, kp_template


def get_image_features(img, orb, desc_template, flann):
    kp_img, desc_img = orb.detectAndCompute(img, None)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    #matches = flann.knnMatch(desc_template, desc_img, k=2)
    matches = bf.match(desc_template, desc_img)
    matches = sorted(matches, key=lambda x: x.distance)
    return matches, kp_img


def get_perspective(matches, kp_template, kp_image, image, template):
    number_of_matches = 2000
    print(f"Number of matches: {len(matches)}")
    source_pts = np.float32([kp_template[m.queryIdx].pt for m in matches[:number_of_matches]]).reshape(-1, 1, 2)
    dest_pts = np.float32([kp_image[m.trainIdx].pt for m in matches[:number_of_matches]]).reshape(-1, 1, 2)
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
        if curImg.shape[0] > 1000 and curImg.shape[1] > 1000:
            curImg = cv2.resize(curImg, (1000, 1000))
        images.append(curImg)
    print(f"{len(images)} Images imported.")
    return images


if __name__ == '__main__':
    images = []
    template = get_template('Resources/citizenship/template.jpg')
    template = noise_removal(template)
    # template = get_template('Resources/license/Homo_template.jpg')
    flann, orb, desc_template, kp_template = get_template_features(template)
    images = get_image('Resources/Citizenship/all/printed/high_quality/FRONT')
    # images = get_image('Resources/license')
    for i in images:
        i = noise_removal(i)
        matches, kp_img = get_image_features(i, orb, desc_template, flann)
        warped_image = get_perspective(matches, kp_template, kp_img, i, template)
        cv2.namedWindow("Original Image", cv2.WINDOW_NORMAL)
        cv2.namedWindow("Warped Image", cv2.WINDOW_NORMAL)
        cv2.imshow("Original Image", i)
        cv2.imshow("Warped Image", warped_image)
        cv2.waitKey(0)
