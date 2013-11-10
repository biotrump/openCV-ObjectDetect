/**
 * @file objectDetection2.cpp
 * @author A. Huaman ( based in the classic facedetect.cpp in samples/c )
 * @brief A simplified version of facedetect.cpp, show how to load a cascade classifier and how to find objects (Face + eyes) in a video stream - Using LBP here
 */
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include <iostream>
#include <stdio.h>

using namespace std;
using namespace cv;

/** Function Headers */
void detectAndDisplay( Mat frame );

/** Global variables */
/*
a--cascade="d:\\repos\\openCV\\win\\opencv\\data\\haarcascades\\haarcascade_frontalface_alt_tree.xml" --nested-cascade="d:\\repos\\openCV\win\\opencv\\data\\haarcascades\\haarcascade_eye.xml" --scale=1.3 -1
*/
//String face_cascade_name = "d:\\repos\\openCV\\win\\opencv\\data\\haarcascades\\haarcascade_frontalface_alt_tree.xml";
String eyes_cascade_name = "d:\\repos\\openCV\\win\\opencv\\data\\haarcascades\\haarcascade_eye.xml";
String face_cascade_name = "d:\\repos\\openCV\\win\\opencv\\data\\lbpcascades\\lbpcascade_frontalface.xml";//lbpcascade_profileface.xml";
//String eyes_cascade_name = "d:\\repos\\openCV\\win\\opencv\\data\\haarcascades\\haarcascade_eye_tree_eyeglasses.xml";
CascadeClassifier face_cascade;
CascadeClassifier eyes_cascade;
string window_name = "Capture - Face detection";

RNG rng(12345);

/**
 * @function main
 */
int main( void )
{
  CvCapture* capture;
  Mat frame;

  //-- 1. Load the cascade
  if( !face_cascade.load( face_cascade_name ) ){ printf("--(!)Error loading\n"); return -1; };
  if( !eyes_cascade.load( eyes_cascade_name ) ){ printf("--(!)Error loading\n"); return -1; };

  //-- 2. Read the video stream
  capture = cvCaptureFromCAM( -1 );
  if( capture )
  {
    for(;;)
    {
      frame = cvQueryFrame( capture );

      //-- 3. Apply the classifier to the frame
      if( !frame.empty() )
       { detectAndDisplay( frame ); }
      else
       { printf(" --(!) No captured frame -- Break!"); break; }

      int c = waitKey(10);
      if( (char)c == 'c' ) { break; }

    }
  }
  return 0;
}

/**
ShowOnlyOneChannelOfRGB
http://stackoverflow.com/questions/14582082/merging-channels-in-opencv
Just create two empty matrix of the same size for Blue and Green.

And you have defined your output matrix as 1 channel matrix. Your output matrix must contain at least 3 channels. 
(Blue, Green and Red). Where Blue and Green will be completely empty and you put your grayscale image as Red channel of the output image. 
*/
void SepShowImgRGB(const string &winName, const vector<Mat> &rgb)
{
    Mat g, fin_img;
    vector<Mat> channels;

    g = Mat::zeros(Size(rgb[2].rows, rgb[2].cols), CV_8UC1);
    channels.push_back(g);	//Blue
    channels.push_back(g);	//Green
    channels.push_back(rgb[2]); //Red
    merge(channels, fin_img);
    imshow("rr", fin_img);
	while (!channels.empty()) channels.pop_back();

	g = g.zeros(Size(rgb[1].rows, rgb[1].cols), CV_8UC1);
    channels.push_back(g);	//Blue
    channels.push_back(rgb[1]);	//Green
    channels.push_back(g); //Red
    merge(channels, fin_img);
    imshow("gg", fin_img);
	while (!channels.empty()) channels.pop_back();

	g = Mat::zeros(Size(rgb[0].rows, rgb[0].cols), CV_8UC1);
    channels.push_back(rgb[0]);	//Blue
    channels.push_back(g);	//Green
    channels.push_back(g); //Red
    merge(channels, fin_img);
    imshow("bb", fin_img);
	while (!channels.empty()) channels.pop_back();
}

void ShowOnlyOneChannelOfRGB(const string &winName, Mat &img)
{
    Mat g, fin_img;
    vector<Mat> channels;

    g = Mat::zeros(Size(img.rows, img.cols), CV_8UC1);

    channels.push_back(g);	//Blue
    channels.push_back(g);	//Green
    channels.push_back(img); //Red

    merge(channels, fin_img);
    imshow(winName, fin_img);
}

/**
 * @function detectAndDisplay
 */
void detectAndDisplay( Mat frame )
{
   std::vector<Rect> faces;
   Mat frame_gray;
   //MAT face_r,face_g,face_b;
   //get the average pixel value of the detected face region in a frame,
   //filter out the pixel value is too black or too white (that means these pixel value is far from mean)
   //calculate the average pixel value again for the detected face.
   cvtColor( frame, frame_gray, CV_BGR2GRAY );
   equalizeHist( frame_gray, frame_gray );

   //-- Detect faces
   face_cascade.detectMultiScale( frame_gray, faces, 1.3, 2, 0, Size(80, 80) );

   for( size_t i = 0; i < faces.size(); i++ )
    {
      Mat faceROI = frame_gray( faces[i] );
      std::vector<Rect> eyes;
	  	Mat faceROI_rgb = frame( faces[i] );
	vector<Mat> rgb;
	Mat ig;
	split(faceROI_rgb, rgb);
	 //split rgb channels
	imshow( "r",rgb[2]);
	imshow( "g",rgb[1]);
	imshow( "b",rgb[0]);
	imshow("face", faceROI_rgb);
   SepShowImgRGB("sep", rgb);
   //integral(rgb[1],ig);
   //computes mean over roi
	cv::Scalar avgPixelIntensity = cv::mean( rgb[0] );//blue
	avgPixelIntensity = cv::mean( rgb[1] );//green
	avgPixelIntensity = cv::mean( rgb[2] );//red
	  //-- Draw the face
      //Point center( faces[i].x + faces[i].width/2, faces[i].y + faces[i].height/2 );
	  Point center( faces[i].x + faces[i].width/2, faces[i].y + faces[i].height/2 + faces[i].height/8 );
	  circle( frame, center, 5, Scalar( 0, 0, 255 ), 3, 8, 0 );
      ellipse( frame, center, Size( faces[i].width/3, faces[i].height/2), 0, 0, 360, Scalar( 255, 0, 0 ), 2, 8, 0 );

      //-- In each face, detect eyes
      eyes_cascade.detectMultiScale( faceROI, eyes, 1.1, 2, 0 |CV_HAAR_SCALE_IMAGE, Size(30, 30) );
	  if( eyes.size() == 2)
      {
         for( size_t j = 0; j < eyes.size(); j++ )
          { //-- Draw the eyes
            Point eye_center( faces[i].x + eyes[j].x + eyes[j].width/2, faces[i].y + eyes[j].y + eyes[j].height/2 );
            int radius = cvRound( (eyes[j].width + eyes[j].height)*0.25 );
            circle( frame, eye_center, radius, Scalar( 255, 0, 255 ), 3, 8, 0 );
          }
       }

    }
   //-- Show what you got
   imshow( window_name, frame );
}
