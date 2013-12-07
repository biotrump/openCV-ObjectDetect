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

#define MAX_FPS					(30)
#define	MAX_SAMPLED_SECONDS		(20)	//6second
#define	MIN_SAMPLED_SECONDS		(3)
#define	MAX_SAMPLED_FRAMES		(MAX_FPS*MAX_SAMPLED_SECONDS)
#define	MIN_SAMPLED_FRAMES		(MAX_FPS*MIN_SAMPLED_SECONDS)

#define	HR_WIN_WIDTH	(640)
#define HR_WIN_HEIGHT	(480)

using namespace std;
using namespace cv;
/*
http://www.cplusplus.com/reference/vector/vector/
std::vector<type T> ==> a vector is a dynamically allocated array,  but an array is static allocation.

*/

/** Global variables */
/*
a--cascade="d:\\repos\\openCV\\win\\opencv\\data\\haarcascades\\haarcascade_frontalface_alt_tree.xml" --nested-cascade="d:\\repos\\openCV\win\\opencv\\data\\haarcascades\\haarcascade_eye.xml" --scale=1.3 -1
*/
#if defined(WIN32) || defined(_WIN32)
//String face_cascade_name = "d:\\repos\\openCV\\win\\opencv\\data\\haarcascades\\haarcascade_frontalface_alt_tree.xml";
String eyes_cascade_name = "d:\\repos\\openCV\\win\\opencv\\data\\haarcascades\\haarcascade_eye.xml";
String face_cascade_name = "d:\\repos\\openCV\\win\\opencv\\data\\lbpcascades\\lbpcascade_frontalface.xml";//lbpcascade_profileface.xml";
#endif

#if defined(__linux__) || defined(LINUX) || defined(__APPLE__) || defined(ANDROID)
String eyes_cascade_name = "../../2.4.7/data/haarcascades/haarcascade_eye.xml";
String face_cascade_name = "../../2.4.7/data/lbpcascades/lbpcascade_frontalface.xml";
#endif

//String eyes_cascade_name = "d:\\repos\\openCV\\win\\opencv\\data\\haarcascades\\haarcascade_eye_tree_eyeglasses.xml";
CascadeClassifier face_cascade;
CascadeClassifier eyes_cascade;
string window_name = "Capture - Face detection";

RNG rng(12345);

#if 0
{
	std::vector<Mat> roi_rgb;//a dynamic matrix array
	split(faceROI_rgb, roi_rgb);
	 //split rgb channels
	imshow( "r",roi_rgb[2]);
	imshow( "g",roi_rgb[1]);
	imshow( "b",roi_rgb[0]);
	imshow("face", faceROI_rgb);
   SepShowImgRGB("sep", roi_rgb);

}
#endif

/**
* @function ShowOnlyOneChannelOfRGB
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

/**
 * @function ShowOnlyOneChannelOfRGB
 */
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
size_t detectAndDisplay( Mat &frame, cv::Scalar &avgPixelIntensity, Rect & roi_new )
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
   face_cascade.detectMultiScale( frame_gray, faces, 1.2, 3, 0, Size(80, 80) );

   //for( size_t i = 0; i < faces.size(); i++ )
   size_t i=0;
   if(faces.size())
    {
      Mat faceROI = frame_gray( faces[i] );
      std::vector<Rect> eyes;
	  	Mat faceROI_rgb = frame( faces[i] );
		roi_new = faces[i];
   //computes mean over roi
   	//Mat m_r(roi_rgb[2]),m_g(roi_rgb[1]),m_b(roi_rgb[0]);//vector to r,g,b matrix
	//avgPixelIntensity = cv::mean( m_g );//mean of g channel mat only
	//http://stackoverflow.com/questions/10959987/equivalent-to-cvavg-in-the-opencv-c-interface
   		avgPixelIntensity = cv::mean( faceROI_rgb );//mean of faceroi , 3 channel matrix (ch0,ch1,ch2)=>(b,g,r)
	//cout << "(" << avgPixelIntensity.val[2] <<", "<< avgPixelIntensity.val[1] <<", "  <<avgPixelIntensity.val[0] << ")"<<endl;

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
	return faces.size();
}


int ProcessFrame(Mat frame, size_t &faces, cv::Scalar & avgPixelIntensity,  Rect  &roi_new)
{
	int64 now_tick=0, t1=cv::getTickCount();
	//double f=cv::getTickFrequency();
	//int fps=0;
	//t = ((double)cv::getTickCount() - t)/cv::getTickFrequency();
	//std::cout<< "FPS@" << 1.0/t  << std::endl;

	faces = detectAndDisplay( frame, avgPixelIntensity, roi_new );

	//fps[?
	now_tick = cv::getTickCount();
	return (int)(now_tick-t1);
	//fps = (int)(f / t);
	//std::cout <<  "FPS@" << fps  << std::endl;
}

/**
 * @function main
 * param -f: face cascade classfier
 *		 -e: eye
 *		 -c: 0,1,2 camera index
 */
int main( int argc, char *argv[] )
{
	VideoCapture vc(0);
  	
  	//-- 1. Load the cascade
  	if( !face_cascade.load( face_cascade_name ) ){ printf("--(!)Error loading\n"); return -1; };
  	if( !eyes_cascade.load( eyes_cascade_name ) ){ printf("--(!)Error loading\n"); return -1; };

  	//-- 2. Read the video stream
	if(vc.isOpened())
  	{
		Mat frame, small_frame;
		unsigned long frame_no=0;
		double now_tick,t1 ;
		double start_tick= (double)cv::getTickCount();
		double maxSampleTicks=cv::getTickFrequency()*(double)MAX_SAMPLED_SECONDS;
		int fps=0;
		int idx=0;
		std::vector<Point3d> Xtr;
		cv::Scalar avgPixelIntensity;//The average/mean pixel value of the face roi's individual RGB channel. (r,g,b,0)
		//vc.set(CV_CAP_PROP_FRAME_WIDTH, 320.0);
		//vc.set(CV_CAP_PROP_FRAME_HEIGHT, 240.0);
		for(;;)
	    {
			Mat avgPixelImage( HR_WIN_HEIGHT, HR_WIN_WIDTH, CV_8UC3, Scalar( 0,0,0) );
	    	size_t nFaces=0;//how many faces are detected
			t1 = (double)cv::getTickCount();
			Rect  roi_new;
			vc >> frame; //get one frame
			frame_no++;
	      
	      	//-- 3. Apply the classifier to the frame
	      	if( !frame.empty() ){
			   //a face is detected last time, so the new detecting area is around the last face ROI to speed up the track
				if(nFaces){
				   int step=roi_new.width /10;
				   (roi_new.x > step)?roi_new.x -=step:0;
				   step = roi_new.height /10;
				   (roi_new.y > step )?roi_new.y-= step:0;
	
					roi_new.width *= 6; // 1.2== 6/5
					roi_new.width /= 5;
					roi_new.height *=  6;//1.2 == 6/5
					roi_new.height /=  5;
					small_frame = frame(roi_new);
					ProcessFrame(small_frame, nFaces, avgPixelIntensity, roi_new);
			   	}
				if(nFaces==0)//first time search, or the new searching area around faceroi fails.
			    	ProcessFrame(frame, nFaces, avgPixelIntensity, roi_new);
				
				if(nFaces>0){			
					//Adding the average point3d  to array
					Xtr.push_back(Point3d(avgPixelIntensity.val[0],avgPixelIntensity.val[1],avgPixelIntensity.val[2]));
					//cout << "vector size" << Xtr.size() << endl;
					idx++;
					cout << "#=" << idx << endl;
				}else{
					//cout <<"out of face " <<endl;
					//idx=0;
					//start_tick= (double)cv::getTickCount();
					//continue;
					//Xtr.clear();
					//goto _waitkey;
				}
			}else{
				printf(" --(!) No captured frame -- Break!");
				//idx=0 ; //reset frame start
				//start_tick= (double)cv::getTickCount(); //reset start of fft
				//Xtr.clear();
				//goto _waitkey;
			}

			now_tick = (double)cv::getTickCount();
			if( (idx  >=  MAX_SAMPLED_FRAMES) || 
				(now_tick - start_tick >= maxSampleTicks) ||
				( !nFaces && (idx >= MIN_SAMPLED_FRAMES ) ) ) 	{//show average HR signal, 
				Scalar     mean;
				Scalar     stddev;
				cv::meanStdDev ( Xtr, mean, stddev );
				
				//raw trace r,g,b : x'[i]=(x[i].[0]-mean.val[0])/stdDev.val[0];
				/// Draw for each channel
				for( int i = 1; i < idx; i++ )
				{
					//raw trace r,g,b : x'[i]=(x[i].[0]-mean.val[0])/stdDev.val[0];
					#define R_RAW_TRACE(ch)		(((double)ch - mean.val[2])/stddev.val[2]*-30.0 + 100.0 )
					#define G_RAW_TRACE(ch) 	(((double)ch - mean.val[1])/stddev.val[1]*-30.0 + 200.0)
					#define B_RAW_TRACE(ch)		(((double)ch - mean.val[0])/stddev.val[0]*-30.0 + 300.0)
					#if 1
					double t0,t1;
						int idxw=1;
						
						if(idx >= (HR_WIN_WIDTH>>1) ) idxw=0;
						else if(idx < (HR_WIN_WIDTH>>2) ) idxw=2;

					t0 = B_RAW_TRACE( Xtr[i-1].x);
					t1 = B_RAW_TRACE(Xtr[i].x);
					line( avgPixelImage, Point( (i-1)<<idxw, B_RAW_TRACE( Xtr[i-1].x) ) ,//b
							   Point( (i)<<idxw,  B_RAW_TRACE(Xtr[i].x) ),
							   Scalar( 255, 0, 0), 1, 8, 0  );
					t0 = G_RAW_TRACE( Xtr[i-1].y);
					t1 = G_RAW_TRACE(Xtr[i].y);
					line( avgPixelImage, Point( (i-1)<<idxw,G_RAW_TRACE(Xtr[i-1].y) ) ,//g
								   Point((i)<<idxw,  G_RAW_TRACE(Xtr[i].y) ),
								   Scalar( 0, 255, 0), 1, 8, 0  );
			  				t0 = R_RAW_TRACE( Xtr[i-1].z);
					t1 = R_RAW_TRACE(Xtr[i].z);
				  	line( avgPixelImage, Point( (i-1)<<idxw, R_RAW_TRACE( Xtr[i-1].z) ) ,//r
								   Point( (i)<<idxw, R_RAW_TRACE( Xtr[i].z) ),
								   Scalar( 0, 0, 255), 1, 8, 0  );
					#else
						
						line( avgPixelImage, Point( (i-1)<<idxw,  Xtr[i-1].x ) ,//b
										   Point( (i)<<idxw,  Xtr[i].x ),
										   Scalar( 255, 0, 0), 1, 8, 0  );
						  line( avgPixelImage, Point( (i-1)<<idxw, Xtr[i-1].y ) ,//g
										   Point((i)<<idxw,  Xtr[i].y ),
										   Scalar( 0, 255, 0), 1, 8, 0  );
						  line( avgPixelImage, Point( (i-1)<<idxw,  Xtr[i-1].z ) ,//r
										   Point( (i)<<idxw,  Xtr[i].z ),
										   Scalar( 0, 0, 255), 1, 8, 0  );
					#endif
				}
				/// Display
				// namedWindow("calcHist Demo", CV_WINDOW_AUTOSIZE );
				imshow("Average RGB channel of FACE ROI", avgPixelImage );
				//cout << "vector size" << Xtr.size()<<endl;
				if(!Xtr.empty()) {
					//cout<<"***"<<endl;
					Xtr.clear();}
			
				frame_no=0;
				start_tick = now_tick;
				//reset i
				idx=0;
			}
			if( !nFaces && (idx < MIN_SAMPLED_FRAMES) ){//drop the HR data				
				Xtr.clear();
				frame_no=0;
				start_tick = now_tick;
				idx=0;
			}
		_waitkey:
		int c = waitKey(10);
	    if( (char)c == 'c' ) { break; }
		}//for
	}
  	return 0;
}

