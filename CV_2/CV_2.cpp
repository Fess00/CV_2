#include <iostream>
#include <opencv2/opencv.hpp>
#include <random>

using namespace std;
using namespace cv;

Mat conv(Mat& img, Mat& filt) {
	int tmpXF = filt.rows;
	int tmpYF = filt.cols;
	int tmpColorCountF = 3;
	int layerF = 5;
	int temp = 1;
	int tmpX = img.rows;
	int tmpY = img.cols;
	int tmpColorCount = 3;
	int X = (tmpX - tmpXF) / temp + 1; 
	int Y = (tmpY - tmpYF) / temp + 1;
	Mat tmp = Mat::zeros(X, Y, CV_32FC3);

	for (int i = 0; i < layerF; i++)
		for (int x = 0; x < X; x++)
			for (int y = 0; y < Y; y++) {
				Vec3b sum = 0;
				for (int k = 0; k < tmpXF; k++)
					for (int l = 0; l < tmpYF; l++)
						for (int m = 0; m < tmpColorCountF; m++)
							sum = sum + img.ptr<Vec3b>(x * temp + k, y * temp + l)[m] * filt.ptr<Vec3b>(k, l)[m][i];
				tmp.ptr<Vec3b>(x, y)[i] = sum;
			}
	imshow("conv", tmp);
	return tmp;
}

Mat ReLu(Mat& img) {
	Mat tmp = img.clone();
	for (int i = 0; i < img.rows; i++)
	{
		for (int j = 0; j < img.cols; j++)
		{
			if (tmp.at<Vec3b>(i, j)[0] > 0)
				continue;
			else
				tmp.at<Vec3b>(i, j)[0] = 0;

			if (tmp.at<Vec3b>(i, j)[1] > 0)
				continue;
			else
				tmp.at<Vec3b>(i, j)[1] = 0;

			if (tmp.at<Vec3b>(i, j)[2] > 0)
				continue;
			else
				tmp.at<Vec3b>(i, j)[2] = 0;
		}
	}
	imshow("ReLu", tmp);
	return tmp;
}

Mat maxPooling(Mat& img) {
	Mat tmp = Mat::zeros(img.size(), CV_32FC3);

	int r = img.rows / 2, c = img.cols / 2;
	tmp.rows = r;
	tmp.cols = c;

	for (int i = 0; i < r; i++)
		for (int j = 0; j < c; j++)
			for (int k = 0; k < 2; k++)
				for (int l = 0; l < 2; l++)
					for (int m = 0; m < 3; m++)
						tmp.ptr<Vec3f>(i, j)[m] = max(img.at<Vec3f>(i * 2 + k, j * 2 + l)[m], tmp.at<Vec3f>(i, j)[m]);

	imshow("maxPooling", tmp);
	return tmp;
}

Mat softMax(Mat& img) {
	Mat tmp;
	float max = *max_element(img.begin<float>(), img.end<float>());
	cv::exp((img - max), tmp);
	float sum = cv::sum(tmp)[0];
	tmp /= sum;

	imshow("softMax", tmp);
	return tmp;
}

int main()
{
	Mat img = imread("D:\\opencv\\sources\\samples\\data\\baboon.jpg");
    Mat f = (Mat_<int>(3, 3));
	srand(time(0));

	for (int i = 0; i < f.rows; i++)
	{
		for (int j = 0; j < f.cols; j++)
		{
			f.at<int>(i, j) = rand() % 1000;
		}
	}

	Mat layer1 = conv(img, f);
	Mat layer2;
	normalize(layer1, layer2, 0, 1, NORM_MINMAX);
	Mat layer3 = ReLu(layer2);
	Mat layer4 = maxPooling(layer3);
	Mat layer5 = softMax(layer4);


	waitKey(0);
	return 0;
}

