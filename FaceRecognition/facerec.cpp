#include "opencv2/core.hpp"
#include "opencv2/face.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgproc/types_c.h"
#include "opencv2/objdetect.hpp"
#include "opencv2/features2d.hpp"
#include <string>
#include <iostream>

#include <fstream>
#ifdef __GNUC__
#include <experimental/filesystem> // Full support in C++17
namespace fs = std::experimental::filesystem;
#else
#include <filesystem>
namespace fs = std::tr2::sys;
#endif

// Author : B00273607
// Based on simplified Eigenfaces tutorial

// ---------------------------------------------------------
// IMPORTANT NOTE: I wasn't sure how to set these in the CMakeLists so they are probably missing. 
// I have these in my configuration properties/debugging

/// Command arguments:
// %OPENCV_ROOT%\sources\data\lbpcascades\lbpcascade_frontalface.xml

/// Environment:
/*
PATH=C:\opencv\build\x64\vc14\bin;%PATH%
OPENCV_ROOT=C:\opencv
*/
// ---------------------------------------------------------

/// Extra features - Trackbar variables
const int kp_slider_max = 145;
int kp_slider = 50;
bool keypoints = false;
/// --------

// Load AT&T faces
void loadFaces(std::vector<cv::Mat> &images, std::vector<int> &labels) {
	// Iterate through all subdirectories, looking for .pgm files
	fs::path p("../att_faces");
	for (const auto &entry : fs::recursive_directory_iterator{ p }) {
		if (fs::is_regular_file(entry.status())) {
			if (entry.path().extension() == ".pgm") {
				std::string str = entry.path().parent_path().stem().string(); // s26 s27 etc.
				int label = atoi(str.c_str() + 1); // s1 -> 1
				images.push_back(cv::imread(entry.path().string().c_str(), 0));
				labels.push_back(label);
			}
		}
	}
	std::cout << "\nImages loaded." << std::endl;
}


void faceRec(cv::Mat frame, cv::CascadeClassifier &lbp_cascade, 
	cv::Ptr<cv::face::BasicFaceRecognizer> model, std::vector<cv::Mat> images) {

	
	cv::Mat original = frame.clone();		// Clone base frame
	cv::Mat gray;
	cvtColor(original, gray, CV_BGR2GRAY);	// Recolour to gray

	std::vector< cv::Rect_<int> > faces;	// Will hold faces found in frame

	const double scale_factor = 1.2;		// how much the image size is reduced (increasing will increase performance)
	const int min_neighbours = 2;			// Minimum number of neighbours to retain a rectangle			
	const cv::Size min_size(46, 56);		// Minimum size to consider when detecting objects (half size of given images)
	const cv::Size max_size(368, 448);		// Max size to consider when detecting objects (close to max size that the screen can fit)

	// Cascade classifier detects objects recognized as faces, returns rectangles containing detected faces
	lbp_cascade.detectMultiScale(gray, faces, scale_factor, min_neighbours, 0, min_size, max_size);

	/// Extra features - Trackbar
	// Trackbar for corner detection treshold
	char TrackbarName[50];
	sprintf(TrackbarName, "Points");
	// Note that once the trackbar is added, it's not going anywhere even if you turn this off
	if(keypoints)
		cv::createTrackbar(TrackbarName, "Face recognition", &kp_slider, kp_slider_max);
	/// --------

	/// Extra features - Corner keypoints
	std::vector < cv::KeyPoint > _kps;
	cv::Ptr < cv::FastFeatureDetector > detector = cv::FastFeatureDetector::create(kp_slider_max - kp_slider);
	detector->detect(frame, _kps);
	if (keypoints)
		drawKeypoints(frame, _kps, frame, cv::Scalar::all(-1), 0);
	/// --------

	// Iterate through faces found by classifier
	for (size_t i = 0; i < faces.size(); i++) {

		cv::Rect face_i = faces[i];
		// Pass in region of interest (rectangle)
		cv::Mat face = gray(face_i);
		cv::Mat face_resized;
		// Resize image to size of samples in folder (size of first image is picked, but they're all 92x112)
		cv::resize(face, face_resized, cv::Size(images[0].cols, images[0].rows), 1.0, 1.0, 2);

		int prediction = 0;
		double confidence = 0;
		// Make class prediction and store confidence of match
		model->predict(face_resized, prediction, confidence);
		
		// Create rectangle around found face (based on rectangles returned by lbp_cascade.detectMultiScale)
		rectangle(frame, face_i, cv::Scalar(191, 189, 59), 1);
		
		std::string box_text;
		prediction == -1 ? box_text = "Face not recognized - confidence too low" 
			: (box_text = cv::format("Prediction = %d | Confidence = %f", prediction, confidence));
		
		/// Extra - Save found match
		if(prediction != -1)
			cv::imwrite("FoundMatch.pgm", face_resized);
		/// ----------------

		// Text position (above rectangle) 
		int topleft_x = std::max(face_i.tl().x - 10, 0);
		int topleft_y = std::max(face_i.tl().y - 10, 0);
		
		// Draw text over rectangle
		cv::putText(frame, box_text, cv::Point(topleft_x, topleft_y), 
			cv::FONT_HERSHEY_PLAIN, 1.0, cv::Scalar(191, 189, 59), 2);
	}
}

int main(int argc, char *argv[])
{
	std::cout << "Face recognition program started." << std::endl;
	std::cout << "(Press 1 during video to enable / disable corner detection)" << std::endl;
	std::vector<cv::Mat> images;
	std::vector<int>     labels;

	// Load images and labels from folder
	loadFaces(images, labels);
	
	// Create cascading classifier (Local binary patterns chosen arbitrarily)
	cv::CascadeClassifier lbp_cascade;
	std::string fn_lbp(argv[1]);
	lbp_cascade.load(fn_lbp);

	// Ensure classifier has been loaded before lengthy training
	assert(!lbp_cascade.empty()); 
	
	// Confidence treshold for face recognizer
	const double threshold = 4200;
	// Face recognizer; A set of Eigenfaces is generated based on Principal Component Analysis (PCA)
	// Could also use FisherFaces, an enhanced version based on Linear Discriminant Analysis (FLDA or LDA) 
	cv::Ptr<cv::face::BasicFaceRecognizer> model = cv::face::createEigenFaceRecognizer(images.size(), threshold);
	// Train face recognizer
	std::cout << "\nNow training..." << std::endl;
	model->train(images, labels);
	std::cout << "Training complete." << std::endl;
	
	// Base frame for video output
	cv::Mat frame;
	double fps = 30;
	const char win_name[] = "Face recognition";

	cv::VideoCapture vid_in(0);   // argument is the camera id
	if (!vid_in.isOpened()) {
		std::cout << "error: Camera 0 could not be opened for capture.\n";
		return -1;
	}

	cv::namedWindow(win_name);

	while (1) {
		vid_in >> frame;
		// Recognize faces in frame
		faceRec(frame, lbp_cascade, model, images);

		int code = (cv::waitKey(1000 / fps) % 255); // how long to wait for a key (msecs)
		if (code == 27) // escape. See http://www.asciitable.com/
			break;
		if (code == 49) // (1) Enable corner detection
			keypoints = !keypoints;	
		// Show frame
		cv::imshow(win_name, frame);
	}

	vid_in.release();
	return 0;
}