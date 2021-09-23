import cv2 as cv
import numpy as np
import pygetwindow

import digital_image_processing as dip


def simple_threshold(image):
    ret, thresh1 = cv.threshold(image, 127, 255, cv.THRESH_BINARY)
    ret, thresh2 = cv.threshold(image, 127, 255, cv.THRESH_BINARY_INV)
    ret, thresh3 = cv.threshold(image, 127, 255, cv.THRESH_TRUNC)
    ret, thresh4 = cv.threshold(image, 127, 255, cv.THRESH_TOZERO)
    ret, thresh5 = cv.threshold(image, 127, 255, cv.THRESH_TOZERO_INV)

    cv.imshow("Binary Threshold", thresh1)
    cv.imshow("Binary Threshold Inverted", thresh2)
    cv.imshow("Truncated Threshold", thresh3)
    cv.imshow("Set to 0", thresh4)
    cv.imshow("Set to 0 Inverted", thresh5)


def adaptive_threshold(image):
    ret, thresh1 = cv.threshold(image, 127, 255, cv.THRESH_BINARY)
    thresh2 = cv.adaptiveThreshold(image, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 11, 2)
    thresh3 = cv.adaptiveThreshold(image, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2)

    cv.imshow("01", thresh1)
    cv.imshow("02", thresh2)
    cv.imshow("03", thresh3)


def otsus_threshold(image):
    # Global thresholding
    ret, thresh1 = cv.threshold(image, 127, 255, cv.THRESH_BINARY)

    # Otsu's thresholding
    ret, thresh2 = cv.threshold(image, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

    # Otsu's thresholding after Gaussian filtering
    blur = cv.GaussianBlur(image, (5, 5), 0)
    ret, thresh3 = cv.threshold(blur, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

    cv.imshow("01", thresh1)
    cv.imshow("02", thresh2)
    cv.imshow("03", thresh3)


def thresholding(image):
    # https://docs.opencv.org/4.5.1/d7/d4d/tutorial_py_thresholding.html
    simple_threshold(image)

    gray_image = cv.imread("../samples/frame_0.png", 0)
    # Grayscale image only
    adaptive_threshold(gray_image)
    otsus_threshold(gray_image)

    cv.waitKey()


if __name__ == '__main__':
    print("Begin")

    # print(pygetwindow.getAllTitles())

    # thresholding()

    # dip.frames_from_window("Celeste", "../samples/processed/")

    # Background subtractor tests - Opencv (MOG2, KNN), Opencv contribute: (GMG, LSBP, CNT, GSOC, MOG)
    # dip.background_subtractor_video_test("../samples/celeste-test.mp4", "CNT")
    dip.background_subtractor_images_test("../samples/processed/", "MOG")

    # @TODO: VGG16 model with Keras - https://www.tensorflow.org/api_docs/python/tf/keras/applications/vgg16/VGG16

    cv.waitKey()

    print("Done")
