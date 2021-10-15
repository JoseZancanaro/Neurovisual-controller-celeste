import cv2 as cv
import numpy as np
import time as timestamp
import os
import matplotlib.pyplot as plt

import imregpoc

from tracker import EuclideanDistTracker
from window_capture import WindowCapture

from skimage.registration import phase_cross_correlation
from skimage.registration._phase_cross_correlation import _upsampled_dft
from scipy.ndimage import fourier_shift


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


def frames_from_window(window_name, samples_path, runtime=5):
    # Initialize the WindowCapture class
    win_cap = WindowCapture(window_name)

    loop_time = timestamp.time()
    loop_end = loop_time + runtime
    count = 0

    # tracker = EuclideanDistTracker()

    # Opencv (MOG2, KNN), Opencv contribute: (GMG, LSBP, CNT, GSOC, MOG)
    # back_sub_type = "MOG2"
    # back_sub = background_subtractor_type(back_sub_type)

    tracker = cv.TrackerCSRT_create()
    bbox = (5, 110, 20, 40)  # native(320x180) - roi(5, 110, 20, 40) - another(5, 60, 30, 100)
    first_frame = True

    while True:  # timestamp.time() < loop_end:
        # Get an updated image of the window
        screenshot = win_cap.get_screenshot()

        # Reduces the captured image, pre-processing and k-means
        native = shrinking(screenshot)
        image = pre_processing(native)
        kmeans = k_means_color_quantization(image)

        # @TODO: applies Madeline tracking
        # back_sub_tracking = back_sub.apply(kmeans)
        # object_tracking = tracking_detection(kmeans, tracker, back_sub)

        # CSRT Tracking tests
        if first_frame:
            # bbox = cv.selectROI(native, False)
            tracker.init(native, bbox)
            first_frame = False

        csrt_tracking_test(native, tracker)

        # Prints
        # cv.imshow("Native resolution", native)
        # cv.imshow("K-means quantization", kmeans)
        # cv.imshow("Madeline tracking ", tracking)

        key = cv.waitKey(30) & 0xff
        if key == 27:
            break

        # Save the captured image
        # cv.imwrite(samples_path + "frame_%d.png" % count, image)
        # count += 1

        # Debug the loop rate
        # print("FPS {}".format(1 / (timestamp.time() - loop_time)))
        # loop_time = timestamp.time()

    cv.destroyAllWindows()


# https://docs.opencv.org/4.5.1/d8/d38/tutorial_bgsegm_bg_subtraction.html
# https://docs.opencv.org/4.5.2/d2/d55/group__bgsegm.html
def background_subtractor_type(bs_type):
    if bs_type == "MOG2":
        back_sub = cv.createBackgroundSubtractorMOG2()
    elif bs_type == "KNN":
        back_sub = cv.createBackgroundSubtractorKNN()
    elif bs_type == "GMG":
        back_sub = cv.bgsegm.createBackgroundSubtractorGMG()
    elif bs_type == "LSBP":
        # Muitos parâmetros, rip
        back_sub = cv.bgsegm.createBackgroundSubtractorLSBP()
    elif bs_type == "CNT":
        back_sub = cv.bgsegm.createBackgroundSubtractorCNT()
    elif bs_type == "GSOC":
        # Muitos parâmetros, rip
        back_sub = cv.bgsegm.createBackgroundSubtractorGSOC()
    else:
        back_sub = cv.bgsegm.createBackgroundSubtractorMOG()

    return back_sub


def background_subtractor_video_test(video_path, bs_type="MOG2"):
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


def background_subtractor_images_test(images_path, bs_type="MOG2"):
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


# https://pysource.com/2021/01/28/object-tracking-with-opencv-and-python/
def tracking_detection(frame, tracker, back_sub):
    height, width, _ = frame.shape

    # Extract Region of interest
    roi = frame[0:height, 0:width]

    # 1. Object Detection
    mask = back_sub.apply(roi)
    # _, mask = cv.threshold(mask, 254, 255, cv.THRESH_BINARY)
    contours, _ = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    detections = []
    for cnt in contours:
        # Calculate area and remove small elements
        area = cv.contourArea(cnt)
        if area > 100:
            # cv2.drawContours(roi, [cnt], -1, (0, 255, 0), 2)
            x, y, w, h = cv.boundingRect(cnt)

            detections.append([x, y, w, h])

    # 2. Object Tracking
    boxes_ids = tracker.update(detections)
    for box_id in boxes_ids:
        x, y, w, h, id = box_id
        cv.putText(roi, str(id), (x, y - 15), cv.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
        cv.rectangle(roi, (x, y), (x + w, y + h), (0, 255, 0), 3)

    cv.imshow("roi", roi)
    # cv.imshow("Frame", frame)
    # cv.imshow("Mask", mask)

    return mask


def object_tracking_images_test(images_path):
    # Take the origin directory, change to images path, sort and restore to origin directory
    initial_path = os.getcwd()
    os.chdir(images_path)
    files_list = sorted(filter(os.path.isfile, os.listdir(".")), key=os.path.getmtime)
    os.chdir(initial_path)

    # Create tracker object
    tracker = EuclideanDistTracker()
    back_sub = background_subtractor_type("MOG2")

    for filename in files_list:
        file = os.path.join(images_path, filename)
        # print(file)

        frame = cv.imread(file)
        # cv.imshow("Images", frame)

        mask = tracking_detection(frame, tracker, back_sub)
        cv.imshow("Mask", mask)

        key = cv.waitKey(30)
        if key == 27:
            break

    cv.waitKey()
    cv.destroyAllWindows()


# https://scikit-image.org/docs/dev/auto_examples/registration/plot_register_translation.html
def image_registration_one(frame):
    image = frame
    # cv.imshow("Grayscale original frame", image)

    shift = (-22.4, 13.32)
    # The shift corresponds to the pixel offset relative to the reference image
    offset_image = fourier_shift(np.fft.fftn(image), shift)
    offset_image = np.fft.ifftn(offset_image)
    print(f'Known offset (y, x): {shift}')

    # pixel precision first
    shift, error, diffphase = phase_cross_correlation(image, offset_image)

    fig = plt.figure(figsize=(8, 3))
    ax1 = plt.subplot(1, 3, 1)
    ax2 = plt.subplot(1, 3, 2, sharex=ax1, sharey=ax1)
    ax3 = plt.subplot(1, 3, 3)

    ax1.imshow(image, cmap='gray')
    ax1.set_axis_off()
    ax1.set_title('Reference image')

    ax2.imshow(offset_image.real, cmap='gray')
    ax2.set_axis_off()
    ax2.set_title('Offset image')

    # Show the output of a cross-correlation to show what the algorithm is
    # doing behind the scenes
    image_product = np.fft.fft2(image) * np.fft.fft2(offset_image).conj()
    cc_image = np.fft.fftshift(np.fft.ifft2(image_product))
    ax3.imshow(cc_image.real)
    ax3.set_axis_off()
    ax3.set_title("Cross-correlation")

    plt.show()

    print(f'Detected pixel offset (y, x): {shift}')

    # subpixel precision
    shift, error, diffphase = phase_cross_correlation(image, offset_image,
                                                      upsample_factor=100)

    fig = plt.figure(figsize=(8, 3))
    ax1 = plt.subplot(1, 3, 1)
    ax2 = plt.subplot(1, 3, 2, sharex=ax1, sharey=ax1)
    ax3 = plt.subplot(1, 3, 3)

    ax1.imshow(image, cmap='gray')
    ax1.set_axis_off()
    ax1.set_title('Reference image')

    ax2.imshow(offset_image.real, cmap='gray')
    ax2.set_axis_off()
    ax2.set_title('Offset image')

    # Calculate the upsampled DFT, again to show what the algorithm is doing
    # behind the scenes.  Constants correspond to calculated values in routine.
    # See source code for details.
    cc_image = _upsampled_dft(image_product, 150, 100, (shift * 100) + 75).conj()
    ax3.imshow(cc_image.real)
    ax3.set_axis_off()
    ax3.set_title("Supersampled XC sub-area")

    plt.show()

    print(f'Detected subpixel offset (y, x): {shift}')


# https://github.com/YoshiRi/ImRegPOC
def image_registration_two(frame, model):
    result = imregpoc.imregpoc(frame, model)
    result.stitching()


# https://www.geeksforgeeks.org/image-registration-using-opencv-python/
def image_registration_three(frame, model):
    # Open the image files.
    img1_color = frame  # Image to be aligned.
    img2_color = model  # Reference image.

    # Convert to grayscale.
    img1 = cv.cvtColor(img1_color, cv.COLOR_BGR2GRAY)
    img2 = cv.cvtColor(img2_color, cv.COLOR_BGR2GRAY)
    height, width = img2.shape

    # Create ORB detector with 5000 features.
    orb_detector = cv.ORB_create(5000)

    # Find keypoints and descriptors.
    # The first arg is the image, second arg is the mask
    # (which is not required in this case).
    kp1, d1 = orb_detector.detectAndCompute(img1, None)
    kp2, d2 = orb_detector.detectAndCompute(img2, None)

    # Match features between the two images.
    # We create a Brute Force matcher with
    # Hamming distance as measurement mode.
    matcher = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)

    # Match the two sets of descriptors.
    matches = matcher.match(d1, d2)

    # Sort matches on the basis of their Hamming distance.
    matches.sort(key=lambda x: x.distance)

    # Take the top 90 % matches forward.
    matches = matches[:int(len(matches) * 0.9)]
    no_of_matches = len(matches)

    # Define empty matrices of shape no_of_matches * 2.
    p1 = np.zeros((no_of_matches, 2))
    p2 = np.zeros((no_of_matches, 2))

    for i in range(len(matches)):
        p1[i, :] = kp1[matches[i].queryIdx].pt
    p2[i, :] = kp2[matches[i].trainIdx].pt

    # Find the homography matrix.
    homography, mask = cv.findHomography(p1, p2, cv.RANSAC)

    # Use this matrix to transform the
    # colored image wrt the reference image.
    transformed_img = cv.warpPerspective(img1_color, homography, (width, height))

    cv.imshow("", transformed_img)
    # Save the output.
    # cv2.imwrite('output.jpg', transformed_img)


# https://learnopencv.com/object-tracking-using-opencv-cpp-python/
def csrt_tracking_test(frame, tracker):
    # Update tracker
    ok, bbox = tracker.update(frame)

    # Draw bounding box
    if ok:
        # Tracking success
        p1 = (int(bbox[0]), int(bbox[1]))
        p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
        cv.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)
        print("safe")
    else:
        # Tracking failure
        cv.putText(frame, "Bucha", (100, 80), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

    # Display tracker type on frame
    # cv.putText(frame, tracker_type + " Tracker", (100, 20), cv.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2);

    # Display FPS on frame
    # cv.putText(frame, "FPS : " + str(int(fps)), (100, 50), cv.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2);

    # Display result
    cv.imshow("Tracking", frame)


# https://github.com/jagracar/OpenCV-python-tests/blob/master/OpenCV-tutorials/featureDetection/fast.py
def fast_feature_detector_test(filename):
    # Grayscale
    img = cv.imread(filename, 0)

    # Initiate FAST object with default values
    fast = cv.FastFeatureDetector_create()  # threshold=25

    # find and draw the keypoints
    kp = fast.detect(img, None)
    img2 = cv.drawKeypoints(img, kp, None, color=(255, 0, 0))

    print("Threshold: ", fast.getThreshold())
    print("nonmaxSuppression: ", fast.getNonmaxSuppression())
    print("neighborhood: ", fast.getType())
    print("Total Keypoints with nonmaxSuppression: ", len(kp))

    cv.imshow("fast_true", img2)

    # Disable nonmaxSuppression
    fast.setNonmaxSuppression(0)
    kp = fast.detect(img, None)

    print("Total Keypoints without nonmaxSuppression: ", len(kp))

    img3 = cv.drawKeypoints(img, kp, None, color=(255, 0, 0))

    cv.imshow("fast_false", img3)
