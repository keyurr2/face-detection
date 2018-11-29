"""

    Created on Wed Nov 28 14:58:11 2018
    
    @author: keyur-r

    This face detector is made using the now classic Histogram of Oriented
    Gradients (HOG) feature combined with a linear classifier, an image
    pyramid, and sliding window detection scheme.  This type of object detector
    is fairly general and capable of detecting many types of semi-rigid objects
    in addition to human faces.  Therefore, if you are interested in making
    your own object detectors then read the train_object_detector.py(dlib) example
    program.
    
    To find faces in image -> python face_detection_hog_svn.py -i <input-image>
    To find faces realtime -> python face_detection_hog_svn.py

"""

from imutils import face_utils
import dlib
import cv2
import argparse
import os


def write_to_disk(image, face_cordinates):
    '''
    This function will save the cropped image from original photo on disk 
    '''
    for (x1, y1, w, h) in face_cordinates:
        cropped_face = image[y1:y1 + h, x1:x1 + w]
        cv2.imwrite(str(y1) + ".jpg", cropped_face)


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


def face_detection_realtime():

    cap = cv2.VideoCapture(0)

    while True:

        # Getting out image by webcam
        _, image = cap.read()

        # Converting the image to gray scale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Get faces into webcam's image
        rects = detector(gray, 0)

        face_cordinates = []
        # For each detected face
        for (i, rect) in enumerate(rects):
            # Finding points for rectangle to draw on face
            x1, y1, x2, y2, w, h = rect.left(), rect.top(), rect.right() + \
                1, rect.bottom() + 1, rect.width(), rect.height()

            # https://stackoverflow.com/questions/46036477/drawing-fancy-rectangle-around-face
            draw_fancy_box(image, (x1, y1), (x2, y2), (127, 255, 255), 2, 10, 20)

            # Drawing simple rectangle around found faces
            # cv2.rectangle(image, (x1, y1), (x1 + w, y1 + h), (0, 255, 0), 2)

            face_cordinates.append((x1, y1, w, h))

            # show the face number
            cv2.putText(image, "Face #{}".format(i + 1), (x1 - 20, y1 - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (51, 51, 255), 2)

        # Show the image
        cv2.imshow("Output", image)

        # To capture found faces from camera
        if cv2.waitKey(30) & 0xFF == ord('s'):
            write_to_disk(image, face_cordinates)

        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    cap.release()


def face_detection(image):

    # Converting the image to gray scale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # The 1 in the second argument indicates that we should upsample the image
    # 1 time.  This will make everything bigger and allow us to detect more
    # faces.

    # Get faces from image
    rects = detector(gray, 1)

    # For each detected face, draw boxes.
    for (i, rect) in enumerate(rects):
        # Finding points for rectangle to draw on face
        x1, y1, x2, y2, w, h = rect.left(), rect.top(), rect.right() + \
            1, rect.bottom() + 1, rect.width(), rect.height()

        # https://stackoverflow.com/questions/46036477/drawing-fancy-rectangle-around-face
        draw_fancy_box(image, (x1, y1), (x2, y2), (127, 255, 255), 2, 10, 20)

        # Drawing simple rectangle around found faces
        # cv2.rectangle(image, (x1, y1), (x1 + w, y1 + h), (0, 255, 0), 2)

        # show the face number
        cv2.putText(image, "Face #{}".format(i + 1), (x1 - 20, y1 - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (51, 51, 255), 2)

    # Show the image
    cv2.imshow("Output", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":

    # Please change your base path
    HOME = "/home/keyur-r/image_data"

    # handle command line arguments
    ap = argparse.ArgumentParser()
    ap.add_argument('-i', '--image', required=False, help='Path to image file')
    args = ap.parse_args()

    # This is based on HOG + SVM classifier
    detector = dlib.get_frontal_face_detector()
    image = None
    if args.image:
        # load input image
        img = os.path.join(HOME, args.image)
        image = cv2.imread(img)

    if image is None:
        print("Real time face detection is starting ... ")
        face_detection_realtime()
    else:
        print("Face detection for image")
        face_detection(image)
