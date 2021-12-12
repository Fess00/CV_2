#include <iostream>
#include <opencv2/opencv.hpp>
#include <random>

using namespace std;
using namespace cv;

Mat conv(Mat& img) {
	Mat fil_1 = (Mat_<int>(3, 3) << 4, 1, 4, 1,2, 3, 1, 2, 4);

	int step = 1;
	int channels = 9, num = 3; 	                      //channels - ��� M
	int h = img.rows, w = img.cols;
	int E = ((h - fil_1.rows) / step + 1);         //F - ������  ��������� �����������
	int F = ((w - fil_1.cols) / step + 1);         //E - ������ ���������

	Mat Out = Mat::zeros(E, F, CV_32FC3);

	for (int m = 0; m < channels; ++m)             //�������� ������
	{
		for (int x = 0; x < E; ++x)				   //������ ��������� �����������
		{
			for (int y = 0; y < F; ++y)			   //������ ��������� �����������
			{
				Vec3f B = 0;

				for (int i = 0; i < fil_1.rows; ++i)
				{
					for (int j = 0; j < fil_1.cols; ++j)
					{
						for (int k = 0; k < num; k++)
						{
							B += img.ptr<Vec3f>(x * step + i, y * step + j)[k] * fil_1.ptr<Vec3f>(i, j)[m][k];
						}
					}
				}
				Out.ptr<Vec3f>(x, y)[m] = B;
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
						tmp.ptr<Vec3f>(i, j)[m] = max(img.at<Vec3f>(i * 2 + k, j * 2 + l)[m], tmp.at<Vec3f>(i, j)[m]);

	imshow("maxPooling", tmp);
	return tmp;
}

Mat softMax(Mat& img) {
	Mat tmp;
	float max = 0.0;
	float sum = 0.0;
	max = *max_element(img.begin<float>(), img.end<float>());
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