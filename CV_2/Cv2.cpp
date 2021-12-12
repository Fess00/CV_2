#include <iostream>
#include <opencv2/opencv.hpp>
#include <random>

using namespace cv;
using namespace std;

Mat conv(Mat& img) {
	Mat fil_1 = (Mat_<int>(3, 3) << 4, 1, 4, 1, 2, 3, 1, 2, 4);

	int step = 1;
	int channels = 9, num = 3; 	                      //channels - это M
	int h = img.rows, w = img.cols;
	int E = ((h - fil_1.rows) / step + 1);         //F - высота  выходного изображени€
	int F = ((w - fil_1.cols) / step + 1);         //E - ширина выходного

	Mat Out = Mat::zeros(E, F, CV_32FC3);

	for (int m = 0; m < channels; ++m)             //¬ыходные каналы
	{
		for (int x = 0; x < E; ++x)				   //¬ысота выходного изображени€
		{
			for (int y = 0; y < F; ++y)			   //Ўирина выходного изображени€
			{
				Vec3i B = 0;

				for (int i = 0; i < fil_1.rows; ++i)
				{
					for (int j = 0; j < fil_1.cols; ++j)
					{
						for (int k = 0; k < num; k++)
						{
							B += img.ptr<Vec3i>(x * step + i, y * step + j)[k] * fil_1.ptr<Vec3i>(i, j)[m][k];
						}
					}
				}
				Out.ptr<Vec3i>(x, y)[m] = B;
			}
		}

	}

	imshow("Image", img);
	imshow("Conv", Out);
	return Out;
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
						tmp.ptr<Vec3i>(i, j)[m] = max(img.at<Vec3i>(i * 2 + k, j * 2 + l)[m], tmp.at<Vec3i>(i, j)[m]);

	imshow("maxPooling", tmp);
	return tmp;
}

Mat softMax(Mat& img) {
	Mat tmp;
	int max = 0.0;
	int sum = 0.0;
	max = *max_element(img.begin<int>(), img.end<int>());
	cv::exp((img - max), tmp);
	sum = cv::sum(tmp)[0];
	tmp /= sum;

	imshow("softMax", tmp);
	return tmp;
}

int main() 
{
	Mat img = imread("D:\\opencv\\sources\\samples\\data\\baboon.jpg");
	
	Mat layer1 = conv(img);
	Mat layer2;
	normalize(layer1, layer2, 0, 1, 32, CV_32FC3);
	imshow("Norm", layer2);
	Mat layer3 = max(layer2, 0);
	imshow("ReLu", layer3);
	Mat layer4 = maxPooling(layer3);
	Mat layer5 = softMax(layer4);

	waitKey(0);
	return 0;
}