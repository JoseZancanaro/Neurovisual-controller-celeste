import cv2 as cv
import numpy as np


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

    # print("Compactness: ", compactness)
    # print("\nLabels: ", labels)
    # print("\nCenters: ", centers)

    # convert back to 8 bit values
    center = np.uint8(centers)

    # flatten the labels array
    label = labels.flatten()

    segmented_image = center[label.flatten()]

    # reshape back to the original image dimension
    segmented_image = segmented_image.reshape(image.shape)

    return segmented_image

