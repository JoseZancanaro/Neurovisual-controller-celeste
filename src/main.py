import cv2 as cv
import pygetwindow
import time as timestamp

from window_capture import WindowCapture
import segment


def image_shrinking(image, scale=3):
    width = int(image.shape[1] / scale)
    height = int(image.shape[0] / scale)
    dimension = (width, height)

    # Resize image: Enlarging (INTER_LINEAR or INTER_CUBIC), shrinking (INTER_AREA)
    return cv.resize(image, dimension, interpolation=cv.INTER_AREA)


def image_pre_processing(image):
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    image = cv.medianBlur(image, 3)

    return image


def frames_from_window(window_name, samples_path, runtime=5):
    # Initialize the WindowCapture class
    win_cap = WindowCapture(window_name)

    loop_time = timestamp.time()
    loop_end = loop_time + runtime
    count = 0

    while timestamp.time() < loop_end:
        # Get an updated image of the window
        screenshot = win_cap.get_screenshot()

        # Reduces the captured image
        image = image_shrinking(screenshot)
        image = image_pre_processing(image)
        image = segment.k_means_color_quantization(image)

        # Save the captured image
        cv.imwrite(samples_path + "frame_%d.png" % count, image)
        count += 1

        # Debug the loop rate
        print("FPS {}".format(1 / (timestamp.time() - loop_time)))
        loop_time = timestamp.time()


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
    frames_from_window("Celeste", "../samples/", 1)

    # thresholding()

    # image = cv.imread("../samples/frame_0.png")
    # cv.imshow("Original", image)

    # img_k3 = segment.k_means_color_quantization(image)
    # cv.imshow("k-means: 3", img_k3)

    # img_k4 = segment.k_means_color_quantization(image, 4)
    # cv.imshow("k-means: 4", img_k4)

    cv.waitKey()

    print("Done")
