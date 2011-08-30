//@todo: aplicar threads:
// http://www.yolinux.com/TUTORIALS/LinuxTutorialPosixThreads.html

/*
 * A Demo to OpenCV Implementation of SURF
 * Further Information Refer to "SURF: Speed-Up Robust Feature"
 * Author: Liu Liu
 * liuliu.1987+opencv@gmail.com
 *
 * modified by: Christian Pfarher
 * c.pfarher@gmail.com
 */

#include <cv.h>
#include <highgui.h>
#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
//#define PI 3.14;
#include <iostream>
#include <vector>

using namespace std;
//christian@christian-laptop:~/Documentos/TESIS/openCV/surfanimado/Debug$ ./main ../../surf/src/Temp/revi.jpg ../../surf/src/pizarron/640-480.jpg

// define whether to use approximate nearest-neighbor search
#define USE_FLANN

int tecla;
IplImage *image = 0;
IplImage* frame;
IplImage* frameDiferencia;
IplImage* frameDiferencia1;

IplImage* framePrev;
IplImage* correspond;
IplImage* roi;
//double areaTreshold=50.0;

double compareSURFDescriptors(const float* d1, const float* d2, double best,
		int length) {
	double total_cost = 0;
	assert( length % 4 == 0 );
	for (int i = 0; i < length; i += 4) {
		double t0 = d1[i] - d2[i];
		double t1 = d1[i + 1] - d2[i + 1];
		double t2 = d1[i + 2] - d2[i + 2];
		double t3 = d1[i + 3] - d2[i + 3];
		total_cost += t0 * t0 + t1 * t1 + t2 * t2 + t3 * t3;
		if (total_cost > best)
			break;
	}
	return total_cost;
}

int naiveNearestNeighbor(const float* vec, int laplacian,
		const CvSeq* model_keypoints, const CvSeq* model_descriptors) {
	int length = (int) (model_descriptors->elem_size / sizeof(float));
	int i, neighbor = -1;
	double d, dist1 = 1e6, dist2 = 1e6;
	CvSeqReader reader, kreader;
	cvStartReadSeq(model_keypoints, &kreader, 0);
	cvStartReadSeq(model_descriptors, &reader, 0);

	for (i = 0; i < model_descriptors->total; i++) {
		const CvSURFPoint* kp = (const CvSURFPoint*) kreader.ptr;
		const float* mvec = (const float*) reader.ptr;
		CV_NEXT_SEQ_ELEM( kreader.seq->elem_size, kreader );
		CV_NEXT_SEQ_ELEM( reader.seq->elem_size, reader );
		if (laplacian != kp->laplacian)
			continue;
		d = compareSURFDescriptors(vec, mvec, dist2, length);
		if (d < dist1) {
			dist2 = dist1;
			dist1 = d;
			neighbor = i;
		} else if (d < dist2)
			dist2 = d;
	}
	if (dist1 < 0.6 * dist2)
		return neighbor;
	return -1;
}

void findPairs(const CvSeq* objectKeypoints, const CvSeq* objectDescriptors,
		const CvSeq* imageKeypoints, const CvSeq* imageDescriptors,
		vector<int>& ptpairs) {
	int i;
	CvSeqReader reader, kreader;
	cvStartReadSeq(objectKeypoints, &kreader);
	cvStartReadSeq(objectDescriptors, &reader);
	ptpairs.clear();

	for (i = 0; i < objectDescriptors->total; i++) {
		const CvSURFPoint* kp = (const CvSURFPoint*) kreader.ptr;
		const float* descriptor = (const float*) reader.ptr;
		CV_NEXT_SEQ_ELEM( kreader.seq->elem_size, kreader );
		CV_NEXT_SEQ_ELEM( reader.seq->elem_size, reader );
		int nearest_neighbor = naiveNearestNeighbor(descriptor, kp->laplacian,
				imageKeypoints, imageDescriptors);
		if (nearest_neighbor >= 0) {
			ptpairs.push_back(i);
			ptpairs.push_back(nearest_neighbor);
		}
	}
}

void flannFindPairs(const CvSeq*,
		const CvSeq* objectDescriptors,
		const CvSeq*,
		const CvSeq* imageDescriptors,
		vector<int>& ptpairs) {

	int length = (int) (objectDescriptors->elem_size / sizeof(float));

	cv::Mat m_object(objectDescriptors->total, length, CV_32F);
	cv::Mat m_image(imageDescriptors->total, length, CV_32F);

	// copy descriptors
	CvSeqReader obj_reader;
	float* obj_ptr = m_object.ptr<float> (0);
	cvStartReadSeq(objectDescriptors, &obj_reader);
	for (int i = 0; i < objectDescriptors->total; i++) {
		const float* descriptor = (const float*) obj_reader.ptr;
		CV_NEXT_SEQ_ELEM( obj_reader.seq->elem_size, obj_reader );
		memcpy(obj_ptr, descriptor, length * sizeof(float));
		obj_ptr += length;
	}
	CvSeqReader img_reader;
	float* img_ptr = m_image.ptr<float> (0);
	cvStartReadSeq(imageDescriptors, &img_reader);
	for (int i = 0; i < imageDescriptors->total; i++) {
		const float* descriptor = (const float*) img_reader.ptr;
		CV_NEXT_SEQ_ELEM( img_reader.seq->elem_size, img_reader );
		memcpy(img_ptr, descriptor, length * sizeof(float));
		img_ptr += length;
	}

	// find nearest neighbors using FLANN
	cv::Mat m_indices(objectDescriptors->total, 2, CV_32S);
	cv::Mat m_dists(objectDescriptors->total, 2, CV_32F);
	cv::flann::Index flann_index(m_image, cv::flann::KDTreeIndexParams(4)); // using 4 randomized kdtrees
	flann_index.knnSearch(m_object, m_indices, m_dists, 2,
			cv::flann::SearchParams(64)); // maximum number of leafs checked

	int* indices_ptr = m_indices.ptr<int> (0);
	float* dists_ptr = m_dists.ptr<float> (0);
	for (int i = 0; i < m_indices.rows; ++i) {
		if (dists_ptr[2 * i] < 0.6 * dists_ptr[2 * i + 1]) {
			ptpairs.push_back(i);
			ptpairs.push_back(indices_ptr[2 * i]);
		}
	}
}

/* a rough implementation for object location */
int locatePlanarObject(const CvSeq* objectKeypoints,
		const CvSeq* objectDescriptors, const CvSeq* imageKeypoints,
		const CvSeq* imageDescriptors, const CvPoint src_corners[4],
		CvPoint dst_corners[4]) {
	double h[9];
	CvMat _h = cvMat(3, 3, CV_64F, h);
	vector<int> ptpairs;
	vector<CvPoint2D32f> pt1, pt2;
	CvMat _pt1, _pt2;
	int i, n;

#ifdef USE_FLANN
	flannFindPairs(objectKeypoints, objectDescriptors, imageKeypoints,
			imageDescriptors, ptpairs);
#else
	findPairs( objectKeypoints, objectDescriptors, imageKeypoints, imageDescriptors, ptpairs );
#endif

	n = ptpairs.size() / 2;
	if (n < 4)
		return 0;

	pt1.resize(n);
	pt2.resize(n);
	for (i = 0; i < n; i++) {
		pt1[i]
				= ((CvSURFPoint*) cvGetSeqElem(objectKeypoints, ptpairs[i * 2]))->pt;
		pt2[i] = ((CvSURFPoint*) cvGetSeqElem(imageKeypoints,
				ptpairs[i * 2 + 1]))->pt;
	}

	_pt1 = cvMat(1, n, CV_32FC2, &pt1[0]);
	_pt2 = cvMat(1, n, CV_32FC2, &pt2[0]);

	if (!cvFindHomography(&_pt1, &_pt2, &_h, CV_RANSAC, 5)) //http://opencv.willowgarage.com/documentation/camera_calibration_and_3d_reconstruction.html#findhomography
		return 0;

	for (i = 0; i < 4; i++) {
		double x = src_corners[i].x, y = src_corners[i].y;
		double Z = 1. / (h[6] * x + h[7] * y + h[8]);
		double X = (h[0] * x + h[1] * y + h[2]) * Z;
		double Y = (h[3] * x + h[4] * y + h[5]) * Z;
		dst_corners[i] = cvPoint(cvRound(X), cvRound(Y));
		/*
		 double rotationX = atan2 (h[0] , h[1]) * 180 / PI;
		 double rotationY = atan2 (h[6] , h[7]) * 180 / PI;
		 double rotationZ = atan2(h[3], h[4])* 180 / PI;;

		 cout<<rotationX<<"        "<<rotationY<<"        "<<rotationZ<<endl;*/

	}

	return 1;
}

/**
 * main function
 *
 * @see http://opencv.willowgarage.com/documentation/feature_detection.html#cvExtractSURF
 */
int main(int argc, char** argv) {
	const char* object_filename = (argc >= 2) ? argv[1]
			: "./src/revibarrio.jpg"; //default object image to search

	double hessianThreshold = (argc >= 3) ? atof(argv[2]) : 1000;

	int enableLines = (argc >= 4) ? atof(argv[3]) : 0; //enable or dissable lines

	CvMemStorage* storage = cvCreateMemStorage(0);

	cvNamedWindow("Object Correspond"); //create window for show image
	//cvNamedWindow("Frames difference"); //create window for show image
	cvNamedWindow("roi");
	static CvScalar colors[] = { { { 0, 0, 255 } }, { { 0, 128, 255 } }, { { 0,
			255, 255 } }, { { 0, 255, 0 } }, { { 255, 128, 0 } }, { { 255, 255,
			0 } }, { { 255, 0, 0 } }, { { 255, 0, 255 } },
			{ { 255, 255, 255 } } };

	IplImage* object1 = cvLoadImage(object_filename, CV_LOAD_IMAGE_GRAYSCALE ); //load object image


	IplImage* object=cvCreateImage(cvSize(object1->width/2,
			object1->height/2), 8, 1); //correspondence image;


	cvSetImageROI(object1, cvRect(0, 0, object1->width/2,
			object1->height/2));

			cvCopy(object1, object);
			cvResetImageROI(object1);





			if (!object) { //bad in arguments:
		fprintf(
				stderr,
				"Can not load %s \n"
					"Usage: ./main [<object_filename>]\n[<hessianThreshold>]\n[<enableLines>]",
				object_filename, hessianThreshold, enableLines);
		exit(-1);
	}

	IplImage* object_keypoints = cvCreateImage(cvGetSize(object), 8, 3);

	cvCvtColor(object, object_keypoints, CV_GRAY2BGR );

	CvSeq *objectKeypoints = 0, *objectDescriptors = 0;
	CvSeq *imageKeypoints = 0, *imageDescriptors = 0;
	int i;

	CvSURFParams params = cvSURFParams(hessianThreshold, 0);

	params.extended = 0;
	params.nOctaveLayers = 2;
	params.nOctaves = 4;

	//double tt = (double) cvGetTickCount();


	cvExtractSURF(object, 0, &objectKeypoints, &objectDescriptors, storage,
			params);
	printf("Object Descriptors: %d\n", objectDescriptors->total);

	//surf points

	for (i = 0; i < objectKeypoints->total; i++) {
		CvSURFPoint* r = (CvSURFPoint*) cvGetSeqElem(objectKeypoints, i);

		//uncomment the follow line for view objectKeypoints information
		//cout<<endl<<"("<<r->pt.x<<", "<<r->pt.y<<")="<<r->size<<" -- dir: "<<r->dir<<" -- hes: "<<r->hessian<<" -- lap: "<<r->laplacian;

		CvPoint center;
		int radius;
		center.x = cvRound(r->pt.x);
		center.y = cvRound(r->pt.y);
		radius = cvRound(r->size * 1.2 / 9. * 2);
		cvCircle(object_keypoints, center, radius, colors[0], 1, 8, 0);
		cvLine(object_keypoints, center, center, colors[3], 2, 8, 0);
	}
	cvShowImage("Object Keypoints", object_keypoints);

	CvCapture *capture = cvCreateCameraCapture(CV_CAP_ANY);
	int ancho = 640, alto = 480;
	cvSetCaptureProperty(capture, CV_CAP_PROP_FRAME_WIDTH, ancho);
	cvSetCaptureProperty(capture, CV_CAP_PROP_FRAME_HEIGHT, alto);

	frame = cvQueryFrame(capture);
	frameDiferencia = cvCreateImage(cvSize(frame->width, frame->height),
			frame->depth, 1);
	framePrev = cvCreateImage(cvSize(frame->width, frame->height),
			frame->depth, 1);
	frameDiferencia1 = cvCreateImage(cvSize(frame->width, frame->height),
			frame->depth, 1);
	image = cvCreateImage(cvSize(frame->width, frame->height), frame->depth, 1);//image over which detect the keypoints (real time detection)
	correspond = cvCreateImage(cvSize(image->width + object->width,
			image->height), 8, 1); //correspondence image

	roi = cvCreateImage(cvSize(image->width + object->width,
				image->height), 8, 1); //correspondence image

	// start and end times
	time_t start, end;
	// fps calculated using number of frames / seconds
	double fps;
	double fpsprom=0;
	double fpsacum=0;
	// frame counter
	int counter = 0;
	// floating point seconds elapsed since start
	double sec;
	// start the clock
#ifdef USE_FLANN
	printf("Using approximate nearest neighbor search\n");
#endif
	time(&start);

	while (true) {
		frame = cvQueryFrame(capture);

		//cout<<"FPS: "<<cvGetCaptureProperty(capture, CV_CAP_PROP_FPS)<<endl; the camera it isn't support this method?

		//cvShowImage("tmp", frame);

		cvCvtColor(frame, image, CV_BGR2GRAY );

		/*BEGIN: show difference frames*/
		//cvAbsDiff(image, framePrev, frameDiferencia); //diferencia frame actual-anterior
		//cvCopy(image, framePrev); //copio el frame actual a un auxiliar para usar en la proxima iteracion

		//cvThreshold(frameDiferencia, frameDiferencia, 50, 255, CV_THRESH_TOZERO);//CV_THRESH_BINARY
		//cvShowImage("Frames difference", frameDiferencia); //muestro el framediferencia

		/*END: show difference frames */

		cvExtractSURF(image, 0, &imageKeypoints, &imageDescriptors, storage,
				params);
		printf("Image Descriptors: %d\n", imageDescriptors->total);


		CvPoint src_corners[4] = { { 0, 0 }, { object->width, 0 }, {
				object->width, object->height }, { 0, object->height } };

		CvPoint dst_corners[4];

		cvSetImageROI(correspond, cvRect(0, 0, object->width, object->height));
		cvCopy(object, correspond);

		/***********************/
		cvSetImageROI(image, cvRect(0, 0, roi->width/2,
								roi->height/2));
		cvSetImageROI(roi, cvRect(0, 0, roi->width/2,
						roi->height/2));
		cvCopy(image, roi);
		cvResetImageROI(image);
		/*************************/
		cvSetImageROI(correspond, cvRect(object->width, 0, correspond->width,
				correspond->height));

		cvCopy(image, correspond);

		cvResetImageROI(correspond);

		if (locatePlanarObject(objectKeypoints, objectDescriptors,
				imageKeypoints, imageDescriptors, src_corners, dst_corners)) {
			//		if (abs(((CvPoint) dst_corners[0]).y-((CvPoint) dst_corners[1]).y)*abs(((CvPoint) dst_corners[0]).x-((CvPoint) dst_corners[3]).x)>areaTreshold) {

			for (i = 0; i < 4; i++) {
				CvPoint r1 = dst_corners[i % 4]; //obtener punto inicial linea
				CvPoint r2 = dst_corners[(i + 1) % 4]; //obtener punto final linea


				cvLine(correspond, cvPoint(r1.x + object->width, r1.y),
						cvPoint(r2.x + object->width, r2.y), colors[8], 10); //hacer linea entre punto inicial y final

				//posicion esquinas del objeto buscado:
				cout << "| " << ((CvPoint) dst_corners[i % 4]).x << "   "
						<< ((CvPoint) dst_corners[i % 4]).y << endl;
				//cout<<"area "<< abs(((CvPoint) dst_corners[0]).y-((CvPoint) dst_corners[1]).y)*abs(((CvPoint) dst_corners[0]).x-((CvPoint) dst_corners[3]).x)<<endl;

			}
		}
		//	} //close area treshold

		vector<int> ptpairs;
#ifdef USE_FLANN
		flannFindPairs(objectKeypoints, objectDescriptors, imageKeypoints,
				imageDescriptors, ptpairs);
#else
		findPairs( objectKeypoints, objectDescriptors, imageKeypoints, imageDescriptors, ptpairs );
#endif

		//linea union puntos corresponedencia:
		if (enableLines) {
			for (i = 0; i < (int) ptpairs.size(); i += 2) {
				CvSURFPoint* r1 = (CvSURFPoint*) cvGetSeqElem(objectKeypoints,
						ptpairs[i]);
				CvSURFPoint* r2 = (CvSURFPoint*) cvGetSeqElem(imageKeypoints,
						ptpairs[i + 1]);
				cvLine(correspond, cvPointFrom32f(r1->pt), cvPoint(cvRound(
						r2->pt.x + object->width), cvRound(r2->pt.y)),
						colors[8]);
			}
		}
		cvShowImage("Object Correspond", correspond);
		cvShowImage("roi", roi);
		//surf points
		/*
		 for (i = 0; i < objectKeypoints->total; i++) {
		 CvSURFPoint* r = (CvSURFPoint*) cvGetSeqElem(objectKeypoints, i);
		 CvPoint center;
		 int radius;
		 center.x = cvRound(r->pt.x);
		 center.y = cvRound(r->pt.y);
		 radius = cvRound(r->size * 1.2 / 9. * 2);
		 cvCircle(object_color, center, radius, colors[0], 1, 8, 0);
		 }
		 cvShowImage("Object", object_color);
		 */

		time(&end);
		// calculate current FPS
		++counter;
		sec = difftime(end, start);
		fps = counter / sec;
		fpsacum += fps;

		fpsprom=(double)((double)fpsacum/(double)counter);
		// will print out Inf until sec is greater than 0
		printf("FPS = %.2f\n", fps);
		printf("FPSProm = %.2f\n", fpsprom);
		// Milisegundos de espera para reconocer la tecla presionada
		tecla = cvWaitKey(1);

		//If ESC key pressed, Key=0x10001B under OpenCV 0.9.7(linux version),
		//remove higher bits using AND operator
		if (((char) tecla & 255) == 27)
			break;
		//cvRelease((void **)&imageKeypoints);
		//cvRelease((void **)&objectKeypoints);
	}
	cvDestroyWindow("Object");
	cvDestroyWindow("Object SURF");
	cvDestroyWindow("Object Correspond");
	cvDestroyWindow("roi");

	return 0;
}
