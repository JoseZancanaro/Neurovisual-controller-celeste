import cv2 as cv
import numpy as np
import time as timestamp

import os
from pathlib import Path

from window_capture import WindowCapture


def shrinking(image, scale=3):
    width = int(image.shape[1] / scale)
    height = int(image.shape[0] / scale)
    dimension = (width, height)

    # Resize image: Enlarging (INTER_LINEAR or INTER_CUBIC), shrinking (INTER_AREA)
    return cv.resize(image, dimension, interpolation=cv.INTER_AREA)


def pre_processing(image):
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    image = cv.medianBlur(image, 3)

    return image


# https://docs.opencv.org/4.5.2/d1/d5c/tutorial_py_kmeans_opencv.html
# https://docs.opencv.org/4.5.2/d5/d38/group__core__cluster.html
def k_means_color_quantization(image, k=3):
    # reshape the image to a 2D array of pixels and 3 color values (RGB)
    pixel_values = image.reshape((-1, 3))
    # convert to float
    pixel_values = np.float32(pixel_values)

    # criteria = (type, max_iteration, epsilon)
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 1.0)
    compactness, labels, centers = cv.kmeans(pixel_values, k, None, criteria, 10, cv.KMEANS_RANDOM_CENTERS)

    print("Compactness: ", compactness)
    print("\nLabels: ", labels)
    print("\nCenters: ", centers)

    # convert back to 8 bit values
    center = np.uint8(centers)

    # flatten the labels array
    label = labels.flatten()

    segmented_image = center[label.flatten()]

    # reshape back to the original image dimension
    segmented_image = segmented_image.reshape(image.shape)

    return segmented_image


def frames_from_window(window_name, samples_path, runtime=5):
    # Initialize the WindowCapture class
    win_cap = WindowCapture(window_name)

    loop_time = timestamp.time()
    loop_end = loop_time + runtime
    count = 0

    while timestamp.time() < loop_end:
        # Get an updated image of the window
        screenshot = win_cap.get_screenshot()

        # Reduces the captured image, pre-processing and k-means
        image = shrinking(screenshot)
        image = pre_processing(image)
        image = k_means_color_quantization(image)

        # @TODO: applies background subtractor

        # Save the captured image
        cv.imwrite(samples_path + "frame_%d.png" % count, image)
        count += 1

        # Debug the loop rate
        print("FPS {}".format(1 / (timestamp.time() - loop_time)))
        loop_time = timestamp.time()


# https://docs.opencv.org/4.5.1/d8/d38/tutorial_bgsegm_bg_subtraction.html
# https://docs.opencv.org/4.5.2/d2/d55/group__bgsegm.html
def background_subtractor_type(bs_type):
    if bs_type == "GMG":
        back_sub = cv.bgsegm.createBackgroundSubtractorGMG()
    elif bs_type == "LSBP":
        back_sub = cv.bgsegm.createBackgroundSubtractorLSBP()
    elif bs_type == "CNT":
        back_sub = cv.bgsegm.createBackgroundSubtractorCNT()
    elif bs_type == "GSOC":
        back_sub = cv.bgsegm.createBackgroundSubtractorGSOC()
    else:
        back_sub = cv.bgsegm.createBackgroundSubtractorMOG()

    return back_sub


def background_subtractor_video_test(video_path, bs_type="MOG"):
    cap = cv.VideoCapture(video_path)

    back_sub = background_subtractor_type(bs_type)

    while True:
        ret, frame = cap.read()
        foreground_mask = back_sub.apply(frame)
        cv.imshow(bs_type, foreground_mask)

        k = cv.waitKey(30) & 0xff
        if k == 27:
            break

    cap.release()
    cv.destroyAllWindows()


def background_subtractor_images_test(images_path, bs_type="MOG"):
    back_sub = background_subtractor_type(bs_type)

    # Take the origin directory, change to images path, sort and restore to origin directory
    initial_path = os.getcwd()
    os.chdir(images_path)
    files_list = sorted(filter(os.path.isfile, os.listdir(".")), key=os.path.getmtime)
    os.chdir(initial_path)

    for filename in files_list:
        file = os.path.join(images_path, filename)
        print(file)

        image = cv.imread(file)
        cv.imshow("Images", image)

        fg_mask = back_sub.apply(image)

        cv.imshow(bs_type, fg_mask)

        k = cv.waitKey(30) & 0xff
        if k == 27:
            break

    cv.destroyAllWindows()