# Face-Detection-Opencv-MTCNN
OpenCV implementation of MTCNN Algorithm

This is an OpenCV implementation of the [MTCNN face detector](https://github.com/ipazc/mtcnn) for Keras in Python3.4+. It is written from scratch, using as a reference the implementation of MTCNN from David Sandberg (FaceNet's MTCNN) in Facenet. It is based on the paper Zhang, K et al. (2016) [ZHANG2016].
<br>
## DEMO:
![ezgif com-gif-maker](https://user-images.githubusercontent.com/37273226/81592405-8d8a6680-93db-11ea-9108-ee8ee26d495c.gif)

Given an input image the MTCNN Model returns the Bounding Box coordinates and other attribute locations.

```python
>>> from mtcnn import MTCNN
>>> import cv2
>>>
>>> img = cv2.cvtColor(cv2.imread("ivan.jpg"), cv2.COLOR_BGR2RGB)
>>> detector = MTCNN()
>>> detector.detect_faces(img)
[
    {
        'box': [277, 90, 48, 63],
        'keypoints':
        {
            'nose': (303, 131),
            'mouth_right': (313, 141),
            'right_eye': (314, 114),
            'left_eye': (291, 117),
            'mouth_left': (296, 143)
        },
        'confidence': 0.99851983785629272
    }
]
```
<br>
These markers are recorded and marked over the WebCam feed using OpenCV library function. This model can detect multiple Faces within its frame. For more information on the model and its benchmarks against various inputs visit [this](https://github.com/ipazc/mtcnn) link.
