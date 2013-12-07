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

#define MAX_FPS		(30)
#define	MAX_SAMPLED_SECONDS		(30)	//6second
#define	MIN_SAMPLED_FRAMES			(MAX_FPS*MAX_SAMPLED_SECONDS)
#define	MIN_SAMPLED_SECONDS		(3)
#define	MAX_SAMPLED_FRAMES		(MAX_FPS*MAX_SAMPLED_SECONDS)

using namespace std;
using namespace cv;
/*
http://www.cplusplus.com/reference/vector/vector/
std::vector<type T> ==> a vector is a dynamically allocated array,  but an array is static allocation.

*/

/** Function Headers */
int ProcessFrame(Mat frame, size_t &faces, cv::Scalar &avgPixelIntensity, 	Rect & roi_new);
size_t detectAndDisplay( Mat &frame,cv::Scalar& avgPixelIntensity, 	Rect  &roi_new );

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

/**
 * @function main
 */
int main( void )
{
	Mat frame;
	VideoCapture vc(0);
  //-- 1. Load the cascade
  if( !face_cascade.load( face_cascade_name ) ){ printf("--(!)Error loading\n"); return -1; };
  if( !eyes_cascade.load( eyes_cascade_name ) ){ printf("--(!)Error loading\n"); return -1; };

  //-- 2. Read the video stream
//  capture = cvCaptureFromCAM( 0 );
//  if( capture )
  if(vc.isOpened())
  {
	Mat frame, small_frame;
	unsigned long frame_no=0;
	double now_tick,t1 ;
	double start_tick= (double)cv::getTickCount();
	double maxSampleTicks=cv::getTickFrequency()*(double)MAX_SAMPLED_SECONDS;
	int fps=0;
	int idx=0;
	Mat matSampledFrames = Mat(1,MAX_SAMPLED_FRAMES, CV_8UC3, cv::Scalar::all(0));//rgb 3 channel, up to 60fps for 10s
	std::vector<Point3d> Xtr;
	cv::Scalar avgPixelIntensity;//The average/mean pixel value of the face roi's individual RGB channel. (r,g,b,0)
	//vc.set(CV_CAP_PROP_FRAME_WIDTH, 320.0);
	//vc.set(CV_CAP_PROP_FRAME_HEIGHT, 240.0);
	for(;;)
    {
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
//				matSampledFrames.at<Vec3b>(0,i)=Point3_<uchar>(i,i,i);	//3D(3 channel) point to matrix element(0,i) which has 3 channels.
				//The first 3 components of Scalar are mean of R,G,B frame
				//Copy the scalar to matrix for later DFT
				//<Vec3b> 3 channel element to matrix element(0,i) which has 3 channels.
				//m(0,i) = Scalar
				//get the average pixel value of indivisual R,G,B channel of the face ROI
				matSampledFrames.at<Vec3b>(0,idx)[0]=(uchar )avgPixelIntensity.val[0];//b
				matSampledFrames.at<Vec3b>(0,idx)[1]=(uchar)avgPixelIntensity.val[1];//g
				matSampledFrames.at<Vec3b>(0,idx)[2]=(uchar)avgPixelIntensity.val[2];//r
			
				//Adding the average point3d  to array
				Xtr.push_back(Point3d(avgPixelIntensity.val[0],avgPixelIntensity.val[1],avgPixelIntensity.val[2]));
				cout << "vector size" << Xtr.size() << endl;
				/*
				printf("(%d,%d,%d)\n",matSampledFrames.at<Vec3b>(0,idx)[2],
					matSampledFrames.at<Vec3b>(0,idx)[1], 
				 matSampledFrames.at<Vec3b>(0,idx)[0]);*/
				idx++;
				cout << "#=" << idx << endl;
			}else{
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
		((now_tick - start_tick)  >= maxSampleTicks) ||
		( (idx > MIN_SAMPLED_FRAMES)&& !nFaces)) 	{
		int hist_w = 600; 
		int hist_h = 400;
		Mat avgPixelImage( hist_h, hist_w, CV_8UC3, Scalar( 0,0,0) );
		Scalar     mean;
		Scalar     stddev;
		cv::meanStdDev ( Xtr, mean, stddev );
		//raw trace r,g,b : x'[i]=(x[i].[0]-mean.val[0])/stdDev.val[0];
		//matSampledFrames.at<Vec3b>(0,i)[0] - mean.val[0]/stddev.val[0];
	
		/// Draw for each channel
		for( int i = 1; i < idx; i++ )
		{
		#if 0
		  //void putText(Mat& img, const string& text, Point org, int fontFace, double fontScale, Scalar color, int thickness=1, int lineType=8, bool bottomLeftOrigin=false )

		#else
		 /*int y= matSampledFrames.at<Vec3b>(0,i-1)[0];
		  line( avgPixelImage, Point( (i-1)<<2,  (matSampledFrames.at<Vec3b>(0,i-1)[0] - 40)*2 ) ,
						   Point( (i)<<2,  (matSampledFrames.at<Vec3b>(0,i)[0]-40)*2 ),
						   Scalar( 255, 0, 0), 1, 8, 0  );
		  line( avgPixelImage, Point( (i-1)<<2,  (matSampledFrames.at<Vec3b>(0,i-1)[1] - 40)*2 ) ,
						   Point((i)<<2,  (matSampledFrames.at<Vec3b>(0,i)[1] - 40)*2 ),
						   Scalar( 0, 255, 0), 1, 8, 0  );
		  line( avgPixelImage, Point( (i-1)<<2,  (matSampledFrames.at<Vec3b>(0,i-1)[2] - 40)*2 ) ,
						   Point( (i)<<2,  (matSampledFrames.at<Vec3b>(0,i)[2] - 40)*2 ),
						   Scalar( 0, 0, 255), 1, 8, 0  );
						   */
	#if 1
			//raw trace r,g,b : x'[i]=(x[i].[0]-mean.val[0])/stdDev.val[0];
	#define R_RAW_TRACE(ch)		(((double)ch - mean.val[2])/stddev.val[2]*-30.0 + 100.0 )
	#define G_RAW_TRACE(ch) 	(((double)ch - mean.val[1])/stddev.val[1]*-30.0 + 200.0)
	#define B_RAW_TRACE(ch)		(((double)ch - mean.val[0])/stddev.val[0]*-30.0 + 300.0)
	#if 1
				double t0,t1;
				t0 = B_RAW_TRACE( Xtr[i-1].x);
				t1 = B_RAW_TRACE(Xtr[i].x);
				line( avgPixelImage, Point( (i-1)<<2, B_RAW_TRACE( Xtr[i-1].x) ) ,//b
						   Point( (i)<<2,  B_RAW_TRACE(Xtr[i].x) ),
						   Scalar( 255, 0, 0), 1, 8, 0  );
				t0 = G_RAW_TRACE( Xtr[i-1].y);
				t1 = G_RAW_TRACE(Xtr[i].y);
		  line( avgPixelImage, Point( (i-1)<<2,G_RAW_TRACE(Xtr[i-1].y) ) ,//g
						   Point((i)<<2,  G_RAW_TRACE(Xtr[i].y) ),
						   Scalar( 0, 255, 0), 1, 8, 0  );
	  				t0 = R_RAW_TRACE( Xtr[i-1].z);
				t1 = R_RAW_TRACE(Xtr[i].z);
		  line( avgPixelImage, Point( (i-1)<<2, R_RAW_TRACE( Xtr[i-1].z) ) ,//r
						   Point( (i)<<2, R_RAW_TRACE( Xtr[i].z) ),
						   Scalar( 0, 0, 255), 1, 8, 0  );
	#else
		line( avgPixelImage, Point( (i-1)<<2,  Xtr[i-1].x ) ,//b
						   Point( (i)<<2,  Xtr[i].x ),
						   Scalar( 255, 0, 0), 1, 8, 0  );
		  line( avgPixelImage, Point( (i-1)<<2, Xtr[i-1].y ) ,//g
						   Point((i)<<2,  Xtr[i].y ),
						   Scalar( 0, 255, 0), 1, 8, 0  );
		  line( avgPixelImage, Point( (i-1)<<2,  Xtr[i-1].z ) ,//r
						   Point( (i)<<2,  Xtr[i].z ),
						   Scalar( 0, 0, 255), 1, 8, 0  );
	#endif
	#endif
	#endif
	  }

	/// Display
	// namedWindow("calcHist Demo", CV_WINDOW_AUTOSIZE );
	imshow("Average RGB channel of FACE ROI  Demo", avgPixelImage );
	cout << "vector size" << Xtr.size()<<endl;
	if(!Xtr.empty()) {
		cout<<"***"<<endl;
		Xtr.clear();}

	frame_no=0;
	start_tick = now_tick;
	//reset i
	idx=0;
	}
_waitkey:
	int c = waitKey(10);
      if( (char)c == 'c' ) { break; }
    }
  }
  return 0;
}

#if 0
void doFFT(void)
{
	  	//t = ((double)cv::getTickCount() - t)/cv::getTickFrequency();
	//std::cout<< "FPS@" << 1.0/t  << std::endl;
	//fps[?
		/*double t=now_tick-t1;
		fps = (int)(f / t); //instant fps
		std::cout <<  "FPS@" << fps  << std::endl;
		std::cout << "Tick:" << (now_tick-start_tick) << "F ticks: " << f << " frame#=" << i << std::endl;*/
		//Do FFT and power spectrum for avgPixelIntensity
		//http://stackoverflow.com/questions/3183078/how-to-implement-1d-fft-filter-for-each-horizontal-line-data-from-image
		//http://docs.opencv.org/doc/tutorials/core/discrete_fourier_transform/discrete_fourier_transform.html?highlight=fourier
		/* 1.Expand the image to an optimal size. The performance of a DFT is dependent of the image size. 
		It tends to be the fastest for image sizes that are multiple of the numbers two, three and five. 
		Therefore, to achieve maximal performance it is generally a good idea to pad border values 
		to the image to get a size with such traits. The getOptimalDFTSize() returns this optimal size and 
		we can use the copyMakeBorder() function to expand the borders of an image:
		*/
		Mat one_row_in_frequency_domain;
		std::vector<Mat> gray_ch;
		//Mat pm_rgb = Mat(1,60,CV_8UC4,avgPixelIntensities);
		//split(avgPixelIntensity->, gray_ch);
		Mat one_row(1, idx	, CV_64FC4);
		// reduce to 1 channel to do fft?
		int n=getOptimalDFTSize(i); //1d matrix //expand input image to optimal size
		Mat padded;
		// on the border add zero pixels for FFT
		copyMakeBorder(one_row, padded, 0, 0, 0, n - i, BORDER_CONSTANT, Scalar::all(0));
		/* 2.Make place for both the complex and the real values. The result of a Fourier Transform is complex. 
		This implies that for each image value the result is two image values (one per component). 
		Moreover, the frequency domains range is much larger than its spatial counterpart. Therefore,
		we store these usually at least in a float format. 
		Therefore we\A1\A6ll convert our input image to this type and expand it with another channel to 
		hold the complex values:
		*/
		Mat planes[] = {Mat_<float>(padded), Mat::zeros(padded.size(), CV_32F)};
		Mat complexI;
		merge(planes, 2, complexI);         // Add to the expanded another plane with zeros
		/* 3. Make the Discrete Fourier Transform. It\A1\A6s possible an in-place calculation (same input as output):
		*/
		cv::dft(complexI, complexI);// this way the result may fit in the source matrix
		/* 4. Transform the real and complex values to magnitude. 
		A complex number has a real (Re) and a complex (imaginary - Im) part. 
		The results of a DFT are complex numbers. The magnitude of a DFT is:
		M^2 = Re(DFT(I))^2 + Im(DFT(I))^2
		*/
		split(complexI, planes);                   // planes[0] = Re(DFT(I), planes[1] = Im(DFT(I))
		magnitude(planes[0], planes[1], planes[0]);// planes[0] = magnitude from planes[0] and planes[1]
		Mat magI = planes[0];
		/* 5. Switch to a logarithmic scale. 
		It turns out that the dynamic range of the Fourier coefficients is too large to be displayed on the screen. 
		We have some small and some high changing values that we can\A1\A6t observe like this. 
		Therefore the high values will all turn out as white points, while the small ones as black. 
		To use the gray scale values to for visualization we can transform our linear scale to a logarithmic one:
		M_1 = \log{(1 + M)}
		*/
		magI += Scalar::all(1);                    // switch to logarithmic scale
		log(magI, magI);
#if 0
		/* 6 Crop and rearrange. Remember, that at the first step, we expanded the image? 
		Well, it\A1\A6s time to throw away the newly introduced values. 
		For visualization purposes we may also rearrange the quadrants of the result, 
		so that the origin (zero, zero) corresponds with the image center.
		*/
		magI = magI(Rect(0, 0, magI.cols & -2, magI.rows & -2));
		int cx = magI.cols/2;
		int cy = magI.rows/2;

		Mat q0(magI, Rect(0, 0, cx, cy));   // Top-Left - Create a ROI per quadrant
		Mat q1(magI, Rect(cx, 0, cx, cy));  // Top-Right
		Mat q2(magI, Rect(0, cy, cx, cy));  // Bottom-Left
		Mat q3(magI, Rect(cx, cy, cx, cy)); // Bottom-Right

		Mat tmp;                           // swap quadrants (Top-Left with Bottom-Right)
		q0.copyTo(tmp);
		q3.copyTo(q0);
		tmp.copyTo(q3);

		q1.copyTo(tmp);                    // swap quadrant (Top-Right with Bottom-Left)
		q2.copyTo(q1);
		tmp.copyTo(q2);
#endif
		/* 7 Normalize. This is done again for visualization purposes. 
		We now have the magnitudes, however this are still out of our image display range of zero to one. 
		We normalize our values to this range using the normalize() function.
		*/
		//normalize(magI, magI, 0, 1, CV_MINMAX); // Transform the matrix with float values into a
                                        // viewable image form (float between values 0 and 1).
		imshow("M FFT", magI);
}
#endif

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
#if 0
		std::vector<Mat> roi_rgb;//a dynamic matrix array
	split(faceROI_rgb, roi_rgb);
	 //split rgb channels
	imshow( "r",roi_rgb[2]);
	imshow( "g",roi_rgb[1]);
	imshow( "b",roi_rgb[0]);
	imshow("face", faceROI_rgb);
   SepShowImgRGB("sep", roi_rgb);
#endif
   //computes mean over roi
   	//Mat m_r(roi_rgb[2]),m_g(roi_rgb[1]),m_b(roi_rgb[0]);//vector to r,g,b matrix
	//avgPixelIntensity = cv::mean( m_g );//mean of g channel mat only
	//http://stackoverflow.com/questions/10959987/equivalent-to-cvavg-in-the-opencv-c-interface
   		avgPixelIntensity = cv::mean( faceROI_rgb );//mean of faceroi , 3 channel matrix (ch0,ch1,ch2)=>(b,g,r)
	cout << "(" << avgPixelIntensity.val[2] <<", "<< avgPixelIntensity.val[1] <<", "  <<avgPixelIntensity.val[0] << ")"<<endl;

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
