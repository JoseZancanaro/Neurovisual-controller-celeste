import cv2 as cv
import pygetwindow
import time as timestamp

from window_capture import WindowCapture


def image_shrinking(image, scale=3):
    width = int(image.shape[1] / scale)
    height = int(image.shape[0] / scale)
    dimension = (width, height)

    # Resize image: Enlarging (INTER_LINEAR or INTER_CUBIC), shrinking (INTER_AREA)
    return cv.resize(image, dimension, interpolation=cv.INTER_AREA)


# TODO: Preprocessing image (particles and background), image segmentation.
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
        screenshot = image_shrinking(screenshot)

        # Save the captured image
        cv.imwrite(samples_path + "frame_%d.png" % count, screenshot)
        count += 1

        # Debug the loop rate
        print("FPS {}".format(1 / (timestamp.time() - loop_time)))
        loop_time = timestamp.time()


if __name__ == '__main__':
    print("Begin")

    print(pygetwindow.getAllTitles())
    frames_from_window("Celeste", "../samples/", 1)

    print("Done")
