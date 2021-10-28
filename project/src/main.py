import cv2 as cv
import pygetwindow

import digital_image_processing as dip
import artificial_intelligence as ai


def neurovisual_control_celeste():
    print("Begin")

    # Print the available applications titles
    print(pygetwindow.getAllTitles())

    # Procedure for capturing and treating the sampling frames
    dip.frames_from_window("Celeste", "../../samples/processed/")

    cv.waitKey()
    print("Done")


def experiments():
    # --Background Subtraction tests - Opencv: (MOG2, KNN), Opencv contribute: (GMG, LSBP, CNT, GSOC, MOG)
    dip.background_subtraction_video_test("../../samples/celeste-test.mp4", "CNT")
    dip.background_subtraction_images_test("../../samples/processed/", "KNN")

    # --Tracking detection test with Euclidean distribution tracker
    dip.tracking_detection_images_test("../../samples/processed/", "KNN")

    # --Image registration tests
    # Only grayscale
    gray_frame = cv.imread("../../samples/native/frame_1.png", 0)
    dip.img_reg_phase_cross_correlation_test(gray_frame)

    # Only grayscale and with a model
    gray_frame = cv.imread("../../samples/native/frame_1.png", 0)
    gray_model = cv.imread("../../samples/madeline_model/model-1.png", 0)
    dip.robust_img_reg_poc_test(gray_frame, gray_model)

    # Same size model, match error :(
    # Different size model, warp error :(
    frame = cv.imread("../../samples/native/frame_1.png")
    model = cv.imread("../../samples/madeline_model/model-1.png")
    dip.img_reg_opencv_test(frame, model)

    # --Fast feature detection test :(
    gray_frame = cv.imread("../../samples/native/frame_1.png", 0)
    dip.fast_feature_detector_test(gray_frame)

    # --Thresholding test
    image = cv.imread("../../samples/native/frame_1.png")
    dip.thresholding(image)

    # VGG16 model test
    model = ai.Vgg16()


if __name__ == '__main__':
    neurovisual_control_celeste()

    # Tests and experiments for this project
    experiments()
