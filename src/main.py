# Credits for https://www.youtube.com/watch?v=WymCpVUPWQ4

import cv2 as cv
from time import time

from window_capture import WindowCapture


# Frames from a window
# @TODO: capture frames from fullscreen application
def frame_capture_from_window(name_window, samples_path):
    # Initialize the WindowCapture class
    win_cap = WindowCapture(name_window)

    loop_time = time()
    count = 0
    while count <= 30:
        # Get an updated image of the window
        screenshot = win_cap.get_screenshot()

        cv.imwrite(samples_path + "frame_%d.png" % count, screenshot)
        count += 1

        # Debug the loop rate
        print("FPS {}".format(1 / (time() - loop_time)))

        loop_time = time()

# Every frame from a video
def frame_capture_from_video(video_path, samples_path):
    video_obj = cv.VideoCapture(video_path)

    if video_obj.isOpened():
        current_frame = 0
        success = 1

        while success:
            success, frame = video_obj.read()

            if success:
                filename = samples_path + "./video/samples/frame_%d.jpg" % current_frame
                print("Creating file... " + filename)

                cv.imwrite(filename, frame)
                current_frame += 1

        video_obj.release()

    cv.destroyAllWindows()

if __name__ == '__main__':
    print("Begin")

    # frame_capture_from_video("./video/test.mp4", "./samples/")
    # print(pygetwindow.getAllTitles())

    frame_capture_from_window("Discord", "../samples/")

    print("Done")