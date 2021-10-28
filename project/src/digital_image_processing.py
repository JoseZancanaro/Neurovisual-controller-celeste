import cv2 as cv
import numpy as np
import time as timestamp
import os
import matplotlib.pyplot as plt

from tracker import EuclideanDistTracker
from window_capture import WindowCapture

from skimage.registration import phase_cross_correlation
from skimage.registration._phase_cross_correlation import _upsampled_dft
from scipy.ndimage import fourier_shift


def shrinking(image, scale=3):
    width = int(image.shape[1] / scale)
    height = int(image.shape[0] / scale)
    dimension = (width, height)

    # Resize image: Enlarging (INTER_LINEAR or INTER_CUBIC), shrinking (INTER_AREA)
    return cv.resize(image, dimension, interpolation=cv.INTER_AREA)


def pre_processing(image):
    # image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    image = cv.medianBlur(image, 3)

    return image


# https://docs.opencv.org/4.5.2/d1/d5c/tutorial_py_kmeans_opencv.html
# https://docs.opencv.org/4.5.2/d5/d38/group__core__cluster.html
def k_means_color_quantization(image, k=3):
    # Reshape the image to a 2D array of pixels and 3 color values (RGB)
    pixel_values = image.reshape((-1, 3))

    # Convert to float
    pixel_values = np.float32(pixel_values)

    # Criteria = (type, max_iteration, epsilon)
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 1.0)
    compactness, labels, centers = cv.kmeans(pixel_values, k, None, criteria, 10, cv.KMEANS_RANDOM_CENTERS)

    # print("Compactness: ", compactness)
    # print("\nLabels: ", labels)
    # print("\nCenters: ", centers)

    # Convert back to 8 bit values
    center = np.uint8(centers)

    # Flatten the labels array
    label = labels.flatten()

    segmented_image = center[label.flatten()]

    # Reshape back to the original image dimension
    segmented_image = segmented_image.reshape(image.shape)

    return segmented_image


# https://learnopencv.com/object-tracking-using-opencv-cpp-python/
def tracking_points(frame, tracker):
    # Update tracker
    success, bbox = tracker.update(frame)

    p1 = (int(bbox[0]), int(bbox[1]))
    p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
    points = [p1, p2]

    return [success, points]


def frames_from_window(window_name, output_path, runtime=20):
    # Initialize the WindowCapture class
    win_cap = WindowCapture(window_name)

    # Runtime control variables
    loop_time = timestamp.time()
    loop_end = loop_time + runtime
    count = 1

    # Tracking variables
    tracker = cv.TrackerCSRT_create()
    bbox = (5, 110, 20, 40)  # native(320x180) - roi(5, 110, 20, 40) - another(5, 60, 30, 100)
    first_frame = None
    is_first_frame = True

    while timestamp.time() < loop_end:
        # Get an updated image of the window
        screenshot = win_cap.get_screenshot()

        # Reduces the captured image, pre-processing and k-means
        native = shrinking(screenshot)
        blur_image = pre_processing(native)
        kmeans = k_means_color_quantization(blur_image)

        # Tracking of the main character
        if is_first_frame:
            # Optional: define a bounty box by mouse
            # mouse_bbox = cv.selectROI(native, False)

            tracker.init(native, bbox)

            first_frame = native.copy()
            is_first_frame = False

        success, (p1, p2) = tracking_points(native, tracker)

        # Draw the tracking in kmeans image copy
        tracking = kmeans.copy()
        if success:
            tracking = cv.rectangle(tracking, p1, p2, (0, 0, 255), 1, 1)
            print("Safe tracking")

        # Press 't' to redefine the tracking with the initial frame and bbox
        redefine_tracking_key = cv.waitKey(30) & 0xff
        if redefine_tracking_key == ord("t"):
            tracker.init(first_frame, bbox)
            print("Redefined tracking")

        # @TODO: Future work: applies vgg16 with the input images

        # Image prints
        cv.imshow("Native resolution", native)
        cv.imshow("Pre-processing", blur_image)
        cv.imshow("K-means quantization", kmeans)
        cv.imshow("Madeline tracking", tracking)

        # If you want save the captured images
        # cv.imwrite(output_path + "frame_%d.png" % count, image)
        # count += 1

        # Debug the loop rate
        print("FPS {}".format(1 / (timestamp.time() - loop_time)))
        loop_time = timestamp.time()

        # Press 'q' to stop
        stop_key = cv.waitKey(30) & 0xff
        if stop_key == ord("q"):
            break

    cv.destroyAllWindows()


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

    cv.imshow("Adaptive 01", thresh1)
    cv.imshow("Adaptive 02", thresh2)
    cv.imshow("Adaptive 03", thresh3)


def otsus_threshold(image):
    # Global thresholding
    ret, thresh1 = cv.threshold(image, 127, 255, cv.THRESH_BINARY)

    # Otsu's thresholding
    ret, thresh2 = cv.threshold(image, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

    # Otsu's thresholding after Gaussian filtering
    blur = cv.GaussianBlur(image, (5, 5), 0)
    ret, thresh3 = cv.threshold(blur, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

    cv.imshow("Otsu's 01", thresh1)
    cv.imshow("Otsu's 02", thresh2)
    cv.imshow("Otsu's 03", thresh3)


# https://docs.opencv.org/4.5.1/d7/d4d/tutorial_py_thresholding.html
def thresholding(image):
    simple_threshold(image)

    # Adaptive and Otsu's use grayscale image
    gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    adaptive_threshold(gray_image)
    otsus_threshold(gray_image)

    cv.waitKey()


def files_list_sort(images_path):
    # Take the origin directory, change to images path, sort and restore to origin directory
    initial_path = os.getcwd()
    os.chdir(images_path)
    files_list = sorted(filter(os.path.isfile, os.listdir(".")), key=os.path.getmtime)
    os.chdir(initial_path)

    return files_list


# https://docs.opencv.org/4.5.1/d8/d38/tutorial_bgsegm_bg_subtraction.html
# https://docs.opencv.org/4.5.2/d2/d55/group__bgsegm.html
def background_subtraction_type(bs_type):
    if bs_type == "MOG2":
        back_sub = cv.createBackgroundSubtractorMOG2()
    elif bs_type == "KNN":
        back_sub = cv.createBackgroundSubtractorKNN()
    elif bs_type == "GMG":
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


# Background subtraction to a video
def background_subtraction_video_test(video_path, bs_type="MOG2"):
    cap = cv.VideoCapture(video_path)

    back_sub = background_subtraction_type(bs_type)

    while True:
        ret, frame = cap.read()

        fg_mask = back_sub.apply(frame)
        cv.imshow(bs_type, fg_mask)

        # Press 'q' to stop
        stop_key = cv.waitKey(30) & 0xff
        if stop_key == ord("q"):
            break

    cap.release()
    cv.destroyAllWindows()


# Background subtraction for a set of images
def background_subtraction_images_test(images_path, bs_type="MOG2"):
    back_sub = background_subtraction_type(bs_type)

    files_list = files_list_sort(images_path)

    for filename in files_list:
        file = os.path.join(images_path, filename)
        print(file)

        image = cv.imread(file)

        fg_mask = back_sub.apply(image)
        cv.imshow(bs_type, fg_mask)

        # Press 'q' to stop
        stop_key = cv.waitKey(30) & 0xff
        if stop_key == ord("q"):
            break

    cv.destroyAllWindows()


# https://pysource.com/2021/01/28/object-tracking-with-opencv-and-python/
def tracking_detection(frame, tracker, back_sub):
    height, width, _ = frame.shape

    # Extract region of interest
    roi = frame[0:height, 0:width]

    # Object detection
    mask = back_sub.apply(roi)
    contours, _ = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    detections = []
    for cnt in contours:
        # Calculate area and remove small elements
        area = cv.contourArea(cnt)
        if area > 100:
            x, y, w, h = cv.boundingRect(cnt)

            detections.append([x, y, w, h])

    # Object tracking
    boxes_ids = tracker.update(detections)
    for box_id in boxes_ids:
        x, y, w, h, id = box_id
        cv.putText(roi, str(id), (x, y - 15), cv.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
        cv.rectangle(roi, (x, y), (x + w, y + h), (0, 255, 0), 3)

    cv.imshow("ROI", roi)

    return mask


# Object tracking detection for a set of images
def tracking_detection_images_test(images_path, bs_type="MOG2"):
    files_list = files_list_sort(images_path)

    # Create tracker and background subtraction
    tracker = EuclideanDistTracker()
    back_sub = background_subtraction_type(bs_type)

    for filename in files_list:
        file = os.path.join(images_path, filename)
        print(file)

        frame = cv.imread(file)

        mask = tracking_detection(frame, tracker, back_sub)
        cv.imshow("Mask", mask)

        # Press 'q' to stop
        stop_key = cv.waitKey(30) & 0xff
        if stop_key == ord("q"):
            break

    cv.destroyAllWindows()


# https://scikit-image.org/docs/dev/auto_examples/registration/plot_register_translation.html
def img_reg_phase_cross_correlation_test(frame):
    # The shift corresponds to the pixel offset relative to the reference image
    shift = (-22.4, 13.32)
    offset_image = fourier_shift(np.fft.fftn(frame), shift)
    offset_image = np.fft.ifftn(offset_image)
    print(f"Known offset (y, x): {shift}")

    # Pixel precision first
    shift, error, diff_phase = phase_cross_correlation(frame, offset_image)

    fig = plt.figure(figsize=(8, 3))
    ax1 = plt.subplot(1, 3, 1)
    ax2 = plt.subplot(1, 3, 2, sharex=ax1, sharey=ax1)
    ax3 = plt.subplot(1, 3, 3)

    ax1.imshow(frame, cmap="gray")
    ax1.set_axis_off()
    ax1.set_title("Reference image")

    ax2.imshow(offset_image.real, cmap="gray")
    ax2.set_axis_off()
    ax2.set_title("Offset image")

    # Show the output of a cross-correlation to show what the algorithm is doing behind the scenes
    image_product = np.fft.fft2(frame) * np.fft.fft2(offset_image).conj()
    cc_image = np.fft.fftshift(np.fft.ifft2(image_product))
    ax3.imshow(cc_image.real)
    ax3.set_axis_off()
    ax3.set_title("Cross-correlation")

    plt.show()

    print(f"Detected pixel offset (y, x): {shift}")

    # Subpixel precision
    shift, error, diff_phase = phase_cross_correlation(frame, offset_image, upsample_factor=100)

    fig = plt.figure(figsize=(8, 3))
    ax1 = plt.subplot(1, 3, 1)
    ax2 = plt.subplot(1, 3, 2, sharex=ax1, sharey=ax1)
    ax3 = plt.subplot(1, 3, 3)

    ax1.imshow(frame, cmap="gray")
    ax1.set_axis_off()
    ax1.set_title("Reference image")

    ax2.imshow(offset_image.real, cmap="gray")
    ax2.set_axis_off()
    ax2.set_title("Offset image")

    # Calculate the upsampled DFT, again to show what the algorithm is doing behind the scenes.
    # Constants correspond to calculated values in routine.
    cc_image = _upsampled_dft(image_product, 150, 100, (shift * 100) + 75).conj()
    ax3.imshow(cc_image.real)
    ax3.set_axis_off()
    ax3.set_title("Supersampled XC sub-area")

    plt.show()

    print(f"Detected subpixel offset (y, x): {shift}")


# See https://github.com/YoshiRi/ImRegPOC to know how apply
def robust_img_reg_poc_test(frame, model):
    # result = imregpoc.imregpoc(frame, model)
    # print(result.getPerspective())
    # result.stitching()
    pass


# https://www.geeksforgeeks.org/image-registration-using-opencv-python/
def img_reg_opencv_test(frame, model):
    img1_color = frame  # Image to be aligned.
    img2_color = model  # Reference image.

    # Convert to grayscale.
    img1 = cv.cvtColor(img1_color, cv.COLOR_BGR2GRAY)
    img2 = cv.cvtColor(img2_color, cv.COLOR_BGR2GRAY)
    height, width = img2.shape

    # Create ORB detector with 5000 features.
    orb_detector = cv.ORB_create(5000)

    # Find keypoints and descriptors.
    kp1, d1 = orb_detector.detectAndCompute(img1, None)
    kp2, d2 = orb_detector.detectAndCompute(img2, None)

    # Match features between the two images.
    # Create a brute force matcher with Hamming distance as measurement mode.
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

    # Use this matrix to transform the colored image wrt the reference image.
    transformed_img = cv.warpPerspective(img1_color, homography, (width, height))

    cv.imshow("Output", transformed_img)


# https://github.com/jagracar/OpenCV-python-tests/blob/master/OpenCV-tutorials/featureDetection/fast.py
def fast_feature_detector_test(frame):
    # Initiate FAST object with default values
    fast = cv.FastFeatureDetector_create()

    # Find and draw the keypoints
    kp = fast.detect(frame, None)
    img2 = cv.drawKeypoints(frame, kp, None, color=(255, 0, 0))

    print("Threshold: ", fast.getThreshold())
    print("NonmaxSuppression: ", fast.getNonmaxSuppression())
    print("Neighborhood: ", fast.getType())
    print("Total Keypoints with nonmaxSuppression: ", len(kp))

    cv.imshow("Fast_true", img2)

    # Disable nonmaxSuppression
    fast.setNonmaxSuppression(0)
    kp = fast.detect(frame, None)

    print("Total Keypoints without nonmaxSuppression: ", len(kp))

    img3 = cv.drawKeypoints(frame, kp, None, color=(255, 0, 0))

    cv.imshow("fast_false", img3)
