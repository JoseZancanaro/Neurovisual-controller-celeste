# Neurovisual Controller Celeste

This project consisted of exploring the application of a neurovisual controller
to the Celeste digital game environment.

During development, game frames were captured, shrunk and later blurred.
Then, they were arranged in the *K-Means clustering* procedure for color quantization.
In addition, *Background Subtraction* algorithms of type KNN, MOG2, CNT, GMG, GSOC, LSBP and MOG have been set up to help with the
character tracking, but the game's camera shake in the scenarios of the Prologue chapter impeded the process. The tracking issue has been solved with the object tracking algorithm from the OpenCV CSRT module,
and it showed promise in both the first and second scenarios of the Prologue.

Unfortunately this study was not able to go into the stages of applying Artificial Intelligence in the game,
due to the effort of the project having been completely focused on the Digital Image Processing of the game frames.

For more information and future work, see the article at: **LINK**  
The set of images referring to the game's frame used in this study can be obtained in: [HERE](https://drive.google.com/drive/folders/1YwSanqiYwS9-Y56eva9azxy7BfaU2H64?usp=sharing)

## Packages

```shell
pip install opencv-python
pip install tensorflow
```

### Optional packages

To get the target application window name
```shell
pip install PyGetWindow
```

To have more background subtraction methods (opencv.bgsegm)
```shell
pip install opencv-contrib-python
```

To realize Image Registration tests
```shell
pip install scikit-image
```