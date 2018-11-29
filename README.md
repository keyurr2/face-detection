Face Detection: Face detection implementation with dlib using HOG + SVM and CNN.
==================

I was just looking for different face detection methods and found a useful library for it called : dlib python bindings

Most of tutorials will provide only one method of face detection which is HOG+SVM based.

While the HOG+SVM based face detector has been around for a while and has gathered a good amount of users I accidentally came across it while browsing through dlib’s github repository.

It's just a bunch of python scripts. Together they:

1. Detect multiple faces using HOG + SVM method in real time as well as from image 
2. Detect multiple faces using CNN method in real time as well as from image
3. Result comparison

The comparison of two different face detectors are as follows :

* Dlib's get_frontal_face : based on HOG + SVM classifier
	- It's so much fast so can be used in real time
	- Unable to detecting faces at odd angles
	- Most of the computer vision engineers use

* Dlib's cnn_face_detection_model_v1 : CNN architecture trained model mmod_human_face_detector.dat
	- It's very poor in speed and computation
	- Far better than dlib's inbuilt model as it's more accurate
	- Less popular

Dataset used to train model : http://dlib.net/files/data/


Installation
==================

To start with project just follow the few steps 

	$ git clone https://github.com/keyurr2/face-detection.git
	$ pip install -r requirements.txt
	$ cd into <project-folder>
	$ python <script-name> -i <image> -w <weightage>

NOTE: How to run each script is mentioned in script itself.

### Screenshot
![HOG+SVM vs CNN for Face Detection](/face_detection_comparison_hog_cnn.png?raw=true "HOG+SVM vs CNN for Face Detection")


Authors
==================

* **Keyur Rathod (keyur.rathod1993@gmail.com)**

License
==================

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details
