#include <iostream>
#include <opencv2/opencv.hpp>
#include <random>
#include <vector>
#include <algorithm>
#include <list>

using namespace cv;
using namespace std;

//random seed gen (from -10 to 10)
random_device rd;
mt19937 gen(rd());
uniform_int_distribution<> dist(-10, 10);


void Conv(Mat img, vector<vector<vector<int>>> core, Mat after) {
	for (int i = 0; i < img.rows - 1; i++)
	{
		for (int j = 0; j < img.cols - 1; j++)
		{
			for (int k = 0; k < img.channels(); k++)
			{
				int tmp;
				if (i == 0 && j != 0) {
					tmp = img.at<Vec3b>(i, j - 1)[k] * core[0][0][k] + img.at<Vec3b>(i, j)[k] * core[0][1][k] + img.at<Vec3b>(i, j + 1)[k] * core[0][2][k] +
						img.at<Vec3b>(i, j - 1)[k] * core[1][0][k] + img.at<Vec3b>(i, j)[k] * core[1][1][k] + img.at<Vec3b>(i, j + 1)[k] * core[1][2][k] +
						img.at<Vec3b>(i + 1, j - 1)[k] * core[2][0][k] + img.at<Vec3b>(i + 1, j)[k] * core[2][1][k] + img.at<Vec3b>(i + 1, j + 1)[k] * core[2][2][k];
					if (tmp > 255)
						tmp = 255;
					else if (tmp < 0)
						tmp = 0;
					after.at<Vec3b>(i, j)[k] = tmp;
				}
				if (i != 0 && j == 0) {
					tmp = img.at<Vec3b>(i - 1, j)[k] * core[0][0][k] + img.at<Vec3b>(i - 1, j)[k] * core[0][1][k] + img.at<Vec3b>(i - 1, j + 1)[k] * core[0][2][k] +
						img.at<Vec3b>(i, j)[k] * core[1][0][k] + img.at<Vec3b>(i, j)[k] * core[1][1][k] + img.at<Vec3b>(i, j + 1)[k] * core[1][2][k] +
						img.at<Vec3b>(i + 1, j)[k] * core[2][0][k] + img.at<Vec3b>(i + 1, j)[k] * core[2][1][k] + img.at<Vec3b>(i + 1, j + 1)[k] * core[2][2][k];
					if (tmp > 255)
						tmp = 255;
					else if (tmp < 0)
						tmp = 0;
					after.at<Vec3b>(i, j)[k] = tmp;
				}
				if (i == 0 && j == 0) {
					tmp = img.at<Vec3b>(i, j)[k] * core[0][0][k] + img.at<Vec3b>(i, j)[k] * core[0][1][k] + img.at<Vec3b>(i, j + 1)[k] * core[0][2][k] +
						img.at<Vec3b>(i, j)[k] * core[1][0][k] + img.at<Vec3b>(i, j)[k] * core[1][1][k] + img.at<Vec3b>(i, j + 1)[k] * core[1][2][k] +
						img.at<Vec3b>(i + 1, j)[k] * core[2][0][k] + img.at<Vec3b>(i + 1, j)[k] * core[2][1][k] + img.at<Vec3b>(i + 1, j + 1)[k] * core[2][2][k];
					if (tmp > 255)
						tmp = 255;
					else if (tmp < 0)
						tmp = 0;
					after.at<Vec3b>(i, j)[k] = tmp;
				}
				if (i == 0 && j == img.cols - 2) {
					tmp = img.at<Vec3b>(i, j - 1)[k] * core[0][0][k] + img.at<Vec3b>(i, j)[k] * core[0][1][k] + img.at<Vec3b>(i, j)[k] * core[0][2][k] +
						img.at<Vec3b>(i, j - 1)[k] * core[1][0][k] + img.at<Vec3b>(i, j)[k] * core[1][1][k] + img.at<Vec3b>(i, j)[k] * core[1][2][k] +
						img.at<Vec3b>(i + 1, j - 1)[k] * core[2][0][k] + img.at<Vec3b>(i + 1, j)[k] * core[2][1][k] + img.at<Vec3b>(i + 1, j)[k] * core[2][2][k];
					if (tmp > 255)
						tmp = 255;
					else if (tmp < 0)
						tmp = 0;
					after.at<Vec3b>(i, j)[k] = tmp;
				}
				if (i != 0 && j == img.cols - 2) {
					tmp = img.at<Vec3b>(i - 1, j - 1)[k] * core[0][0][k] + img.at<Vec3b>(i - 1, j)[k] * core[0][1][k] + img.at<Vec3b>(i - 1, j)[k] * core[0][2][k] +
						img.at<Vec3b>(i, j - 1)[k] * core[1][0][k] + img.at<Vec3b>(i, j)[k] * core[1][1][k] + img.at<Vec3b>(i, j)[k] * core[1][2][k] +
						img.at<Vec3b>(i + 1, j - 1)[k] * core[2][0][k] + img.at<Vec3b>(i + 1, j)[k] * core[2][1][k] + img.at<Vec3b>(i + 1, j)[k] * core[2][2][k];
					if (tmp > 255)
						tmp = 255;
					else if (tmp < 0)
						tmp = 0;
					after.at<Vec3b>(i, j)[k] = tmp;
				}
				if (i == img.rows - 2 && j == img.cols - 2) {
					tmp = img.at<Vec3b>(i - 1, j - 1)[k] * core[0][0][k] + img.at<Vec3b>(i - 1, j)[k] * core[0][1][k] + img.at<Vec3b>(i - 1, j)[k] * core[0][2][k] +
						img.at<Vec3b>(i, j - 1)[k] * core[1][0][k] + img.at<Vec3b>(i, j)[k] * core[1][1][k] + img.at<Vec3b>(i, j)[k] * core[1][2][k] +
						img.at<Vec3b>(i, j - 1)[k] * core[2][0][k] + img.at<Vec3b>(i, j)[k] * core[2][1][k] + img.at<Vec3b>(i, j)[k] * core[2][2][k];
					if (tmp > 255)
						tmp = 255;
					else if (tmp < 0)
						tmp = 0;
					after.at<Vec3b>(i, j)[k] = tmp;
				}
				if (i == img.rows - 2 && j != 0) {
					tmp = img.at<Vec3b>(i - 1, j - 1)[k] * core[0][0][k] + img.at<Vec3b>(i - 1, j)[k] * core[0][1][k] + img.at<Vec3b>(i - 1, j + 1)[k] * core[0][2][k] +
						img.at<Vec3b>(i, j - 1)[k] * core[1][0][k] + img.at<Vec3b>(i, j)[k] * core[1][1][k] + img.at<Vec3b>(i, j + 1)[k] * core[1][2][k] +
						img.at<Vec3b>(i, j - 1)[k] * core[2][0][k] + img.at<Vec3b>(i, j)[k] * core[2][1][k] + img.at<Vec3b>(i, j + 1)[k] * core[2][2][k];
					if (tmp > 255)
						tmp = 255;
					else if (tmp < 0)
						tmp = 0;
					after.at<Vec3b>(i, j)[k] = tmp;
				}
				if (i == img.rows - 2 && j == 0) {
					tmp = img.at<Vec3b>(i - 1, j)[k] * core[0][0][k] + img.at<Vec3b>(i - 1, j)[k] * core[0][1][k] + img.at<Vec3b>(i - 1, j + 1)[k] * core[0][2][k] +
						img.at<Vec3b>(i, j)[k] * core[1][0][k] + img.at<Vec3b>(i, j)[k] * core[1][1][k] + img.at<Vec3b>(i, j + 1)[k] * core[1][2][k] +
						img.at<Vec3b>(i, j)[k] * core[2][0][k] + img.at<Vec3b>(i, j)[k] * core[2][1][k] + img.at<Vec3b>(i, j + 1)[k] * core[2][2][k];
					if (tmp > 255)
						tmp = 255;
					else if (tmp < 0)
						tmp = 0;
					after.at<Vec3b>(i, j)[k] = tmp;
				}
				if ((i != 0 && j != 0) && (i != img.rows - 2 && j != img.cols - 2)) {
					tmp = img.at<Vec3b>(i - 1, j - 1)[k] * core[0][0][k] + img.at<Vec3b>(i - 1, j)[k] * core[0][1][k] + img.at<Vec3b>(i - 1, j + 1)[k] * core[0][2][k] +
						img.at<Vec3b>(i, j - 1)[k] * core[1][0][k] + img.at<Vec3b>(i, j)[k] * core[1][1][k] + img.at<Vec3b>(i, j + 1)[k] * core[1][2][k] +
						img.at<Vec3b>(i + 1, j - 1)[k] * core[2][0][k] + img.at<Vec3b>(i + 1, j)[k] * core[2][1][k] + img.at<Vec3b>(i + 1, j + 1)[k] * core[2][2][k];
					if (tmp > 255)
						tmp = 255;
					else if (tmp < 0)
						tmp = 0;
					after.at<Vec3b>(i, j)[k] = tmp;
				}
			}
		}
	}
	imshow("Convolution", after);
}

void MinMaxNorm(Mat src) {
	for (int i = 0; i < src.rows; i++)
	{
		for (int j = 0; j < src.cols; j++)
		{
			for (int k = 0; k < 3; k++)
			{
				src.at<Vec3b>(i, j)[0] = (src.at<Vec3b>(i, j)[0] - 0) / 255;
			}
		}
	}

	imshow("Norm", src);
 }

void Relu(Mat src) {
	for (int i = 0; i < src.rows; i++)
	{
		for (int j = 0; j < src.cols; j++)
		{
			for (int k = 0; k < 3; k++)
			{
				if (src.at<Vec3b>(i, j)[k] > 0)
					continue;
				else
					src.at<Vec3b>(i, j)[k] = 0;
			}
		}
	}

	imshow("Relu", src);
}

uchar Max(uchar x, uchar y, uchar z, uchar w) {
	int arr[4] = { static_cast<int>(x), static_cast<int>(y), static_cast<int>(z), static_cast<int>(w) };
	int max = 0;
	for (int i = 0; i < 4; i++)
	{
		if (max < arr[i])
			max = arr[i];
	}

	return max;
}

void MaxPooling(Mat src, Mat after) {
	int newWidth = src.cols / 2;
	int newHeight = src.rows / 2;
	int r = 0, c = 0;

	for (int i = 0; i < newHeight; i++)
	{
		for (int j = 0; j < newWidth; j++)
		{
			for (int k = 0; k < 3; k++)
			{
				after.at<Vec3b>(i, j)[k] = Max(src.at<Vec3b>(i * 2, j * 2)[k], src.at<Vec3b>(i * 2, (j * 2) + 1)[k],
					src.at<Vec3b>((i * 2) + 1, j * 2)[k], src.at<Vec3b>((i * 2) + 1, (j * 2) + 1)[k]);
			}
		}
	}

	imshow("MaxPooling", after);
}

void SoftMax(Mat src, Mat dst)
{
	int z = 0;
	for (int i = 0; i < src.rows; i++)
	{
		double val1 = 0.0;
		double val2 = 0.0;
		double val3 = 0.0;
		for (int j = 0; j < src.cols; j++)
		{
			for (int ki = 0; ki < src.rows; ki++)
			{
				for (int kj = 0; kj < src.cols; kj++)
				{
					val1 += exp(src.at<Vec3b>(ki, kj)[0]);
					val2 += exp(src.at<Vec3b>(ki, kj)[1]);
					val3 += exp(src.at<Vec3b>(ki, kj)[2]);
					if (kj == src.cols - 1 && ki == src.rows - 1) {
						dst.at<Vec3d>(i, j)[0] = exp(src.at<Vec3b>(i, j)[0]) / val1;
						dst.at<Vec3d>(i, j)[1] = exp(src.at<Vec3b>(i, j)[1]) / val2;
						dst.at<Vec3d>(i, j)[2] = exp(src.at<Vec3b>(i, j)[2]) / val3;
					}
				}
			}
		}
		cout << ++z << endl;
	}
}

int main() 
{
	//Load image
	Mat src = imread("D:\\opencv\\sources\\samples\\data\\baboon.jpg");
	//init filter 3x3x3 for conv | (core)
	vector<vector<vector<int>>> core;
	core.resize(3);
	for (int i = 0; i < 3; i++)
	{
		core[i].resize(3);
		for (int j = 0; j < 3; j++)
		{
			core[i][j].resize(3);
		}
	}
	//randomize all segments in 3d matrix (core) in range from -10 to 10
	for (int i = 0; i < 3; i++)
	{
		for (int j = 0; j < 3; j++)
		{
			for (int k = 0; k < 3; k++)
			{
				core[i][j][k] = dist(gen);
			}
		}
	}
	// image after conv
	Mat afterConv;
	// copy image
	src.copyTo(afterConv);

	Conv(src, core, afterConv);
	MinMaxNorm(afterConv);
	Relu(afterConv);

	//image after MaxPooling
	int newH = afterConv.rows / 2;
	int newW = afterConv.cols / 2;
	cout << newH << endl << newW;
	Mat afterPooling;
	afterPooling = Mat::zeros(newH, newW, CV_8UC3);

	MaxPooling(afterConv, afterPooling);

	//image after SoftMax
	Mat afterSoft;
	afterSoft = Mat::zeros(newH, newW, CV_64FC3);
	cout << afterPooling.size;
	cout << afterSoft.size;

	SoftMax(afterPooling, afterSoft);

	waitKey(0);
	return 0;
}