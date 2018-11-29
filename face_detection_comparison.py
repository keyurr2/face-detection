"""

    Created on Wed Nov 28 16:46:15 2018
    
    @author: keyur-r

    To find faces in image -> python face_detection_hog_svn.py -i <input-image>
    To find faces realtime -> python face_detection_hog_svn.py -i <input-image> -w <weight>

"""

from imutils import face_utils
import dlib
import cv2
import argparse
import os
import time


def write_to_disk(image, face_cordinates):
    '''
    This function will save the cropped image from original photo on disk 
    '''
    for (x1, y1, w, h) in face_cordinates:
        cropped_face = image[y1:y1 + h, x1:x1 + w]
        cv2.imwrite(str(y1) + ".jpg", cropped_face)


def face_detection(image):

    # Converting the image to gray scale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # The 1 in the second argument indicates that we should upsample the image
    # 1 time.  This will make everything bigger and allow us to detect more
    # faces.

    # Get faces from image
    start = time.time()
    faces_hog = hog_face_detector(gray, 1)
    end = time.time()
    print("HOG : ", format(end - start, '.2f'))

    # apply face detection (cnn)
    start = time.time()
    faces_cnn = cnn_face_detector(image, 1)
    end = time.time()
    print("CNN : ", format(end - start, '.2f'))

    # For each detected face, draw boxes.

    # HOG + SVN
    for (i, face) in enumerate(faces_hog):
        # Finding points for rectangle to draw on face
        x, y, w, h = face.left(), face.top(), face.width(), face.height()

        # Drawing simple rectangle around found faces
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)

    # CNN
    for (i, face) in enumerate(faces_cnn):
        # Finding points for rectangle to draw on face
        x, y, w, h = face.rect.left(), face.rect.top(), face.rect.width(), face.rect.height()

        # Drawing simple rectangle around found faces
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # write at the top left corner of the image
    # for color identification
    img_height, img_width = image.shape[:2]
    cv2.putText(image, "Dlib HOG + SVN", (img_width - 200, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (0, 0, 255), 2)
    cv2.putText(image, "Dlib CNN", (img_width - 200, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (0, 255, 0), 2)

    # Show the image
    cv2.imshow("Output", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":

    # Please change your base path
    HOME = "/home/keyur-r/image_data"

    # handle command line arguments
    ap = argparse.ArgumentParser()
    ap.add_argument('-w', '--weights', help='Path to Weights',
                    default='./mmod_human_face_detector.dat')
    ap.add_argument('-i', '--image', required=True, help='Path to image file')
    args = ap.parse_args()

    # initialize hog + svm based face detector
    hog_face_detector = dlib.get_frontal_face_detector()

    # initialize cnn based face detector with the weights
    cnn_face_detector = dlib.cnn_face_detection_model_v1(args.weights)

    image = None
    if args.image:
        # load input image
        img = os.path.join(HOME, args.image)
        image = cv2.imread(img)

    if image is None:
        print("Please provide image ...")
    else:
        print("Face detection for image")
        face_detection(image)
