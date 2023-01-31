import cv2

from align_image.sift_detector import get_template_features, get_image_features, get_aligned_img
from utils.imutil import read_image, get_image
from image_processing.image_processing import noise_removal
from ocr.easy_ocr import extract_text_easyocr
# from spell_correction.pySpell import spell_corrector as pySpellcheck
from key_value_mapping.KV_map import text_clean, find_value

if __name__ == '__main__':
    template = read_image('../test_images/Homo_template.jpg')
    flann, sift, desc_template, kp_template = get_template_features(template)
    images = get_image('../test_images/licenseDataset')
    for i in images:
        matches, kp_img = get_image_features(i, sift, desc_template, flann)
        warped_image, keyed_img = get_aligned_img(matches, kp_template, kp_img, i, template)
        noiseless_img = noise_removal(warped_image)
        text, boxed_img = extract_text_easyocr(noiseless_img)
        print("__________Details____________")
        w_string = ""
        for bbox, txt, conf in text:
            w_string = w_string + ' ' + txt
        print(w_string)
        cleaned_text = text_clean(w_string)
        find_value(cleaned_text)
        print("____________________________")
        # print("____________With spell correction________________")
        # w_string = ""
        # for bbox, txt, conf in text:
        #    w_string = w_string + ' ' + txt
        # pySpellcheck(w_string)
        # print("____________________________")
        cv2.namedWindow("Text Detection", cv2.WINDOW_NORMAL)
        cv2.imshow("Text Detection", boxed_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
