# Revisar, créditos para https://www.youtube.com/watch?v=WymCpVUPWQ4

import cv2 as cv
import os
from time import time
import pygetwindow

from window_capture import WindowCapture

def frame_capture_from_window(name_window, directory):

    # Change the working directory to the folder this script is in.
    # Doing this because I'll be putting the files from each video in their own folder on GitHub
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    # initialize the WindowCapture class
    wincap = WindowCapture(name_window)

    loop_time = time()
    count = 0
    while(True):

        # get an updated image of the game
        screenshot = wincap.get_screenshot()

        #qqcv.imshow('Computer Vision', screenshot)
        cv.imwrite(directory + "frame_%d.png" %count, screenshot)
        count += 1

        # debug the loop rate
        print('FPS {}'.format(1 / (time() - loop_time)))

        loop_time = time()

        # press 'q' with the output window focused to exit.
        # waits 1 ms every loop to process key presses
        #if cv.waitKey(1) & 0xFF == ord('q'):
            #cv.destroyAllWindows()
            #break

    print('Done.')

"""
import cv2

def frame_capture_from_video(path):
    video_obj = cv2.VideoCapture(path)

    if video_obj.isOpened():
        current_frame = 0
        success = 1

        while success:
            success, frame = video_obj.read()

            if success:
                filename = "./video/samples/frame_%d.jpg" %current_frame
                print("Creating file... " + filename)

                cv2.imwrite(filename, frame)
                current_frame += 1

        video_obj.release()

    cv2.destroyAllWindows()

def frame_capture_from_application():
    pass

"""

if __name__ == '__main__':
    #frame_capture_from_video("./video/test.mp4")
    titles = pygetwindow.getAllTitles()
    print(titles)

    frame_capture_from_window("Discord", "../samples/")