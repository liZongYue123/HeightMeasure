//
//  main.cpp.
//  depth from focus.
//
//  Created by lizongyue on 2024/04/01.
//

#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/ximgproc/weighted_median_filter.hpp>
#include "GCoptimization.h"

#define IMG_SIZE 107
#define MAX_ITER 10
#define SMOOTH_CONST 1
#define IMG_READ_PATH "F:/3D/DepthFromFocus/DepthFromFocus/Origin/"
#define IMG_WRITE_PATH "F:/3D//DepthFromFocus/DepthFromFocus/Result/"


using namespace std;
using namespace cv;

void img_read(string img_path, Mat* img_stack);
void align_images(Mat* imgStack, Mat* resultStack);
void calculate_costvolume(Mat* imgStack, Mat* costVolume);
void all_in_focus(Mat* img_stack, Mat &depth_map, Mat &result_map);
void image_refinement(const Mat* costVolume, Mat &result_img);
void show_colorized_image(Mat source, string name);
void save_image(Mat source, string name);

int main() 
{
	Mat img_stack_original[IMG_SIZE];
	Mat img_stack_aligned[IMG_SIZE];
	Mat costVolume[IMG_SIZE];
	Mat resultImg;
	Mat depthImg;
	Mat depthImg_refined;
	img_read(IMG_READ_PATH, img_stack_original);
	align_images(img_stack_original, img_stack_aligned);
	calculate_costvolume(img_stack_aligned, costVolume);
	image_refinement(costVolume, depthImg_refined);
	all_in_focus(img_stack_aligned, depthImg_refined, resultImg);
	imshow("result", resultImg);
	save_image(resultImg, "result");
	waitKey(0);
	return 0;
}

void img_read(string img_path, Mat* img_stack) 
{
	cout << "����ͼ��...";
	for (int i = 0; i < IMG_SIZE; i++) 
	{
		if (i < 9)
			img_stack[i] = imread(img_path + "0" + to_string(i + 1) + ".bmp");
		else
			img_stack[i] = imread(img_path + to_string(i + 1) + ".bmp");

		if (img_stack[i].empty()) 
		{
			printf("����ͼ��ʧ��\n");
			exit(-1);
		}
	}
	cout << "���!" << endl;
}

void all_in_focus(Mat* img_stack, Mat &depth_map, Mat &result_map) 
{
	result_map = Mat(depth_map.size().width, depth_map.size().height, CV_8UC3);
	for (int x = 0; x < depth_map.size().width; x++) {
		for (int y = 0; y < depth_map.size().height; y++) {
			result_map.at<Vec3b>(y, x) = img_stack[depth_map.at<uchar>(y, x)].at<Vec3b>(y, x);
		}
	}
}

void image_refinement(const Mat* costVolume, Mat &result_img) 
{
	cout << "�������ͼ...";
	const int width = costVolume[0].size().width;
	const int height = costVolume[0].size().height;
	result_img = Mat(width, height, CV_8UC1);
	try {
		GCoptimization *gc = new GCoptimizationGridGraph(width, height, IMG_SIZE);
		for (int x = 0; x < width; x++) 
		{
			for (int y = 0; y < height; y++) 
			{
				for (int l = 0; l < IMG_SIZE; l++) 
				{
					gc->setDataCost(x + height * y, l, IMG_SIZE - costVolume[l].at<uchar>(y, x));
				}
			}
		}
		for (int l1 = 0; l1 < IMG_SIZE; l1++)
		{
			for (int l2 = 0; l2 < IMG_SIZE; l2++)
			{
				int cost = abs(l1 - l2);
				gc->setSmoothCost(l1, l2, cost * SMOOTH_CONST);
			}
		}
		cout << "�ع��ü�ͼ��...";
		gc->expansion(MAX_ITER);
		for (int x = 0; x < width; x++) 
		{
			for (int y = 0; y < height; y++)
			{
				result_img.at<uchar>(y, x) = gc->whatLabel(x + height * y);
			}
		}
		ximgproc::weightedMedianFilter(result_img, result_img, result_img, 2);
		show_colorized_image(result_img, "������");
		delete gc;
	}
	catch (GCException e) 
	{
		e.Report();
	}
	/*
	Mat colorMap;
	normalize(result_img, colorMap, 0, 255, NORM_MINMAX);
	applyColorMap(colorMap, colorMap, COLORMAP_RAINBOW);
	imshow("refined depth", colorMap);
	*/
	cout << "���!" << endl;
}

void calculate_costvolume(Mat* imgStack, Mat* costVolume) 
{
	cout << "�����ʼ���ͼ...";
	Mat originalImg[IMG_SIZE];
	for (int i = 0; i < IMG_SIZE; i++) 
	{
		cvtColor(imgStack[i], originalImg[i], COLOR_BGR2GRAY);
		//GaussianBlur(originalImg[i], originalImg[i], Size(3,3), 0);
		Laplacian(imgStack[i], costVolume[i], CV_16S, 3, 1);
		convertScaleAbs(costVolume[i], costVolume[i]);
	}
	save_image(costVolume[0], "cost volume 1");
	save_image(costVolume[106], "cost volume 107");
	//�������ɳ�ʼ���ͼ����������
	Mat depthMap = Mat(imgStack[0].size().width, imgStack[0].size().height, CV_8UC1);
	for (int x = 0; x < imgStack[0].size().width; x++) 
	{
		for (int y = 0; y < imgStack[0].size().height; y++) 
		{
			int currentIndex = 0;
			int maxIntensity = -1;
			for (int index = 0; index < IMG_SIZE; index++) 
			{
				if (maxIntensity < (int)costVolume[index].at<uchar>(y, x))
				{
					maxIntensity = (int)costVolume[index].at<uchar>(y, x);
					currentIndex = index;
				}
			}
			depthMap.at<uchar>(y, x) = currentIndex;
		}
	}
	//����������
	show_colorized_image(depthMap, "��ʼ���ͼ");
	cout << "���!" << endl;
}

void align_images(Mat* imgStack, Mat* resultStack) 
{
	cout << "����ͼ��...";
	Mat templateImg;
	cvtColor(imgStack[0], templateImg, COLOR_BGR2GRAY);
	resultStack[0] = imgStack[0];
	std::vector<KeyPoint> keypointTemplate;
	Mat descriptorTemplate;
	Ptr<Feature2D> orb = ORB::create(CPU_MAX_FEATURE);
	orb->detectAndCompute(templateImg, Mat(), keypointTemplate, descriptorTemplate);
	std::vector<DMatch> matches;
	std::vector<KeyPoint> keypointLast = keypointTemplate;
	Mat descriptorLast = descriptorTemplate;
	Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");
	Mat currentImage;
	std::vector<KeyPoint> keypointCurrent;
	Mat descriptorCurrent;
	Mat lastHomography;
	std::vector<Point2f> points1, points2;
	for (int i = 1; i < IMG_SIZE; i++) 
	{
		cvtColor(imgStack[i], currentImage, COLOR_BGR2GRAY);
		orb->detectAndCompute(currentImage, Mat(), keypointCurrent, descriptorCurrent);
		matcher->match(descriptorLast, descriptorCurrent, matches, Mat());
		for (size_t i = 0; i < matches.size(); i++) 
		{
			points1.push_back(keypointLast[matches[i].queryIdx].pt);
			points2.push_back(keypointCurrent[matches[i].trainIdx].pt);
		}
		Mat h = findHomography(points1, points2, RANSAC);
		lastHomography = i == 1 ? h : lastHomography * h;
		keypointLast = keypointCurrent;
		descriptorLast = descriptorCurrent;
		warpPerspective(imgStack[i], resultStack[i], lastHomography, imgStack[0].size());
	}
	show_colorized_image(imgStack[0] - imgStack[29], "����ǰ");
	show_colorized_image(imgStack[0] - resultStack[29], "������");
	cout << "���!" << endl;
}

void show_colorized_image(Mat source, string name) 
{
	Mat colorMap;
	normalize(source, colorMap, 0, 255, NORM_MINMAX);
	applyColorMap(colorMap, colorMap, COLORMAP_RAINBOW);
	imshow(name, colorMap);
	save_image(colorMap, name);
}

void save_image(Mat source, string name) 
{
	imwrite(IMG_WRITE_PATH + name + ".png", source);
}