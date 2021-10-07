import cv2 as cv
import pygetwindow

import digital_image_processing as dip
import artificial_intelligence as ai


if __name__ == '__main__':
    print("Begin")

    # print(pygetwindow.getAllTitles())

    # dip.thresholding(image)

    dip.frames_from_window("Celeste", "../samples/processed/")

    ## Background subtractor tests - Opencv (MOG2, KNN), Opencv contribute: (GMG, LSBP, CNT, GSOC, MOG)
    # dip.background_subtractor_video_test("../samples/celeste-test.mp4", "CNT")
    # dip.background_subtractor_images_test("../samples/processed/", "KNN")

    ## Object tracking test
    # dip.object_tracking_images_test("../samples/processed/")

    ## Image registration tests
    # Only grayscale
    # frame = cv.imread("../samples/native/frame_1.png", 0)
    # dip.image_registration_one(frame)

    # Only grayscale and with a model
    # frame = cv.imread("../samples/native/frame_1.png", 0)
    # model = cv.imread("../samples/madeline_model/model-1.png", 0)
    # dip.image_registration_two(frame, model)

    ## modelo de mesmo tamanho, erro no match.
    ## modelo de tamanha diferente, erro no warp
    # frame = cv.imread("../samples/native/frame_1.png")
    # model = cv.imread("../samples/madeline_model/model-2.png")
    # dip.image_registration_three(frame, model)

    # @TODO: VGG16 model
    # model = ai.Vgg16()

    cv.waitKey()
    print("Done")
