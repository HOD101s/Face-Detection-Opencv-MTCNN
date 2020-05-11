import matplotlib.pyplot as plt
from mtcnn.mtcnn import MTCNN
import numpy as np
import cv2

cap = cv2.VideoCapture(0)
while True:
    ret,img = cap.read()
    detector = MTCNN()
    faces = detector.detect_faces(img)
    for face in faces:
        box = face['box']
        cv2.rectangle(img, (box[0],box[1]), (box[0]+box[2],box[1]+box[3]), (255, 255, 255), 2)
        cv2.circle(img,face['keypoints']['left_eye'] , 1, (255, 0, 0), 10)
        cv2.circle(img,face['keypoints']['right_eye'] , 1, (0, 0, 255), 10)
        cv2.circle(img,face['keypoints']['nose'] , 1, (0, 0, 0), 10)
        cv2.circle(img,face['keypoints']['mouth_left'] , 1, (0, 0, 0), 10)
        cv2.circle(img,face['keypoints']['mouth_right'] , 1, (0, 0, 0), 10)
        cv2.putText(img,str(round(face['confidence'],5)),(box[0]-5,box[1]-5),cv2.FONT_HERSHEY_SIMPLEX,0.5,(250,250,0),2,cv2.LINE_AA)
    cv2.imshow('Face Detection',img)
    if cv2.waitKey(1) & 0xff == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()
