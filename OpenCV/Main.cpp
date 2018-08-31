#include <opencv2/opencv.hpp>
#include "opencv2/features2d.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/stitching.hpp"

using namespace cv;

int main(int argc, char** argv) {

	Mat img1 = imread(argv[1], -1);
	Mat img2 = imread(argv[2], -1);
	Mat key1, key2, key3, img1_rotated, desc1, desc2;

	if (img1.empty()) return -1;
	if (img2.empty()) return -1;

	Point2f src_center(img1.cols / 2.0F, img1.rows / 2.0F);

	Mat rot_mat = getRotationMatrix2D(src_center, 0, 1.0);
	
	warpAffine(img1, img1_rotated, rot_mat, img1.size());

	Ptr<ORB> detector = ORB::create(250);

	std::vector<KeyPoint> kp1, kp2, kp3;

	detector->detectAndCompute(img1_rotated, Mat(), kp1, desc1);
	detector->detectAndCompute(img2, Mat(), kp2, desc2);

	Ptr<DescriptorMatcher> matcher = makePtr<FlannBasedMatcher>(makePtr<flann::LshIndexParams>(12, 20, 2));
	
	std::vector< DMatch > matches;
	
	matcher->match(desc1, desc2, matches);

	double max_dist = 0; double min_dist = 100;
	
	for (int i = 0; i < matches.size(); i++)
	{
		double dist = matches[i].distance;
		if (dist < min_dist) min_dist = dist;
		if (dist > max_dist) max_dist = dist;
	}
	
	std::vector< DMatch > good_matches;
	
	for (int i = 0; i < matches.size(); i++)
	{
		if (matches[i].distance <= max(2 * min_dist, 0.02))
		{
			good_matches.push_back(matches[i]);
		}
	}

	std::vector<Point2f> obj;
	std::vector<Point2f> scene;

	Mat img_matches, warpedImg;
	
	drawMatches(img1_rotated, kp1, img2, kp2, good_matches, img_matches, Scalar::all(-1), Scalar::all(-1), std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
	
	imshow("Good Matches", img_matches);
	
	for (int i = 0; i < (int)good_matches.size(); i++)
	{
		printf("-- Good Match [%d] Keypoint 1: %d  -- Keypoint 2: %d  \n", i, good_matches[i].queryIdx, good_matches[i].trainIdx);
		obj.push_back(kp1[good_matches[i].queryIdx].pt);
		scene.push_back(kp2[good_matches[i].trainIdx].pt);
	}

	Mat H = findHomography(scene,obj,RANSAC);

	Size sz;
	
	sz.width = img1.size().width+ img2.size().width;
	sz.height = img2.size().height;

	warpPerspective(img2, warpedImg, H, sz);
	
	imshow("warped", warpedImg);

	Mat final(Size(img2.cols * 2 + img2.cols, img2.rows * 2), CV_8UC3);

	Mat roi1(final, Rect(0, 0, img1.cols, img1.rows));
	Mat roi2(final, Rect(0, 0, warpedImg.cols, warpedImg.rows));

	warpedImg.copyTo(roi2);
	img1.copyTo(roi1);

	imshow("Result", final);
	
	cv::waitKey(0);

	return 0;
}