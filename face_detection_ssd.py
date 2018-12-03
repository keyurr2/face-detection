"""

    Created on Sun Dec 2 20:54:11 2018
    
    @author: keyur-r

    SSD pretrained caffe model based face detection using it with opencv's dnn module.
    (https://docs.opencv.org/3.4.0/d5/de7/tutorial_dnn_googlenet.html)
    
    python face_detection_ssd.py -p <prototxt> -m <caffe-model> -t <thresold>

"""

from imutils import face_utils, video
import dlib
import cv2
import argparse
import os
import numpy as np


def draw_fancy_box(img, pt1, pt2, color, thickness, r, d):
    '''
    To draw some fancy box around founded faces in stream
    '''
    x1, y1 = pt1
    x2, y2 = pt2

    # Top left
    cv2.line(img, (x1 + r, y1), (x1 + r + d, y1), color, thickness)
    cv2.line(img, (x1, y1 + r), (x1, y1 + r + d), color, thickness)
    cv2.ellipse(img, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, thickness)

    # Top right
    cv2.line(img, (x2 - r, y1), (x2 - r - d, y1), color, thickness)
    cv2.line(img, (x2, y1 + r), (x2, y1 + r + d), color, thickness)
    cv2.ellipse(img, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, thickness)

    # Bottom left
    cv2.line(img, (x1 + r, y2), (x1 + r + d, y2), color, thickness)
    cv2.line(img, (x1, y2 - r), (x1, y2 - r - d), color, thickness)
    cv2.ellipse(img, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, thickness)

    # Bottom right
    cv2.line(img, (x2 - r, y2), (x2 - r - d, y2), color, thickness)
    cv2.line(img, (x2, y2 - r), (x2, y2 - r - d), color, thickness)
    cv2.ellipse(img, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, thickness)


def find_faces(img, detections):
    total_faces = 0

    # Draw boxes around found faces
    for i in range(0, detections.shape[2]):
        # Probability of prediction
        prediction_score = detections[0, 0, i, 2]
        if prediction_score < args.thresold:
            continue
        # Finding height and width of frame
        (h, w) = img.shape[:2]
        # compute the (x, y)-coordinates of the bounding box for the
        # object
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (x1, y1, x2, y2) = box.astype("int")

        total_faces += 1

        prediction_score_str = "{:.2f}%".format(prediction_score * 100)

        label = "Face #{} ({})".format(total_faces, prediction_score_str)

        # https://stackoverflow.com/questions/46036477/drawing-fancy-rectangle-around-face
        draw_fancy_box(img, (x1, y1), (x2, y2), (127, 255, 255), 2, 10, 20)
        # show the face number with prediction score
        cv2.putText(img, label, (x1 - 20, y1 - 20),
                    cv2.FONT_HERSHEY_TRIPLEX, 0.6, (51, 51, 255), 2)

    # show the output frame
    cv2.imshow("Face Detection with SSD", img)


def face_detection_realtime():

    # Feed from computer camera with threading
    cap = video.VideoStream(src=0).start()

    while True:

        # Getting out image frame by webcam
        img = cap.read()

        # https://docs.opencv.org/trunk/d6/d0f/group__dnn.html#ga29f34df9376379a603acd8df581ac8d7
        inputBlob = cv2.dnn.blobFromImage(cv2.resize(
            img, (300, 300)), 1, (300, 300), (104, 177, 123))

        detector.setInput(inputBlob)
        detections = detector.forward()
        find_faces(img, detections)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()
    cap.stop()

if __name__ == "__main__":

    # handle command line arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--prototxt", default="./deploy.prototxt.txt",
                    help="Caffe 'deploy' prototxt file")
    ap.add_argument("-m", "--model", default="./res10_300x300_ssd_iter_140000.caffemodel",
                    help="Pre-trained caffe model")
    ap.add_argument("-t", "--thresold", type=float, default=0.6,
                    help="Thresold value to filter weak detections")
    args = ap.parse_args()

    # This is based on SSD deep learning pretrained model
    detector = cv2.dnn.readNetFromCaffe(args.prototxt, args.model)

    print("Real time face detection is starting ... ")
    face_detection_realtime()
