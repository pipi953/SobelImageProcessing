#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>
#include <device_functions.h>

#include <opencv2\opencv.hpp>
#include <opencv2/core/core.hpp> 
#include<opencv2/highgui/highgui.hpp>  

#include <iostream>
using namespace std;
using namespace cv;

//Sobel���ӱ�Ե���˺���
__global__ void sobelInCuda(unsigned char *dataIn, unsigned char *dataOut, int imgHeight, int imgWidth)
{
    int xIndex = threadIdx.x + blockIdx.x * blockDim.x;
    int yIndex = threadIdx.y + blockIdx.y * blockDim.y;
    int index = yIndex * imgWidth + xIndex;
    int Gx = 0;
    int Gy = 0;

    if (xIndex > 0 && xIndex < imgWidth - 1 && yIndex > 0 && yIndex < imgHeight - 1)
    {
        Gx = dataIn[(yIndex - 1) * imgWidth + xIndex + 1] + 2 * dataIn[yIndex * imgWidth + xIndex + 1] + dataIn[(yIndex + 1) * imgWidth + xIndex + 1]
            - (dataIn[(yIndex - 1) * imgWidth + xIndex - 1] + 2 * dataIn[yIndex * imgWidth + xIndex - 1] + dataIn[(yIndex + 1) * imgWidth + xIndex - 1]);
        Gy = dataIn[(yIndex - 1) * imgWidth + xIndex - 1] + 2 * dataIn[(yIndex - 1) * imgWidth + xIndex] + dataIn[(yIndex - 1) * imgWidth + xIndex + 1]
            - (dataIn[(yIndex + 1) * imgWidth + xIndex - 1] + 2 * dataIn[(yIndex + 1) * imgWidth + xIndex] + dataIn[(yIndex + 1) * imgWidth + xIndex + 1]);
        dataOut[index] = (abs(Gx) + abs(Gy)) / 2;
    }
}

//Sobel���ӱ�Ե���CPU����
void sobel(Mat srcImg, Mat dstImg, int imgHeight, int imgWidth)
{
    int Gx = 0;
    int Gy = 0;
    for (int i = 1; i < imgHeight - 1; i++)
    {
        uchar *dataUp = srcImg.ptr<uchar>(i - 1);
        uchar *data = srcImg.ptr<uchar>(i);
        uchar *dataDown = srcImg.ptr<uchar>(i + 1);
        uchar *out = dstImg.ptr<uchar>(i);
        for (int j = 1; j < imgWidth - 1; j++)
        {
            Gx = (dataUp[j + 1] + 2 * data[j + 1] + dataDown[j + 1]) - (dataUp[j - 1] + 2 * data[j - 1] + dataDown[j - 1]);
            Gy = (dataUp[j - 1] + 2 * dataUp[j] + dataUp[j + 1]) - (dataDown[j - 1] + 2 * dataDown[j] + dataDown[j + 1]);
            out[j] = (abs(Gx) + abs(Gy)) / 2;
        }
    }
}

int main()
{
   Mat grayImg = imread("1.jpg", 0);
    
   if (!grayImg.data) {
       std::cout << "Image not loaded";
       return -1;
   }

   namedWindow("windows", CV_WINDOW_AUTOSIZE);
   imshow("windows", grayImg);


    imwrite("Gray_Image.jpg", grayImg);


    int imgHeight = grayImg.rows;
    int imgWidth = grayImg.cols;

    Mat gaussImg;
    //��˹�˲�
    GaussianBlur(grayImg, gaussImg, Size(3, 3), 0, 0, BORDER_DEFAULT);

    //Sobel����CPUʵ��
    Mat dst(imgHeight, imgWidth, CV_8UC1, Scalar(0));
    sobel(gaussImg, dst, imgHeight, imgWidth);

    //CUDAʵ�ֺ�Ĵ��ص�ͼ��
    Mat dstImg(imgHeight, imgWidth, CV_8UC1, Scalar(0));

    //����GPU�ڴ�
    unsigned char *d_in;
    unsigned char *d_out;

    cudaMalloc((void**)&d_in, imgHeight * imgWidth * sizeof(unsigned char));
    cudaMalloc((void**)&d_out, imgHeight * imgWidth * sizeof(unsigned char));

    //����˹�˲����ͼ���CPU����GPU
    cudaMemcpy(d_in, gaussImg.data, imgHeight * imgWidth * sizeof(unsigned char), cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(32, 32);
    dim3 blocksPerGrid((imgWidth + threadsPerBlock.x - 1) / threadsPerBlock.x, (imgHeight + threadsPerBlock.y - 1) / threadsPerBlock.y);

    //���ú˺���
    sobelInCuda << < blocksPerGrid, threadsPerBlock >> > (d_in, d_out, imgHeight, imgWidth);

    //��ͼ�񴫻�GPU
    cudaMemcpy(dstImg.data, d_out, imgHeight * imgWidth * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    imwrite("dst_Image.jpg", dst);

    imwrite("dstImg_Image.jpg", dstImg);

    //�ͷ�GPU�ڴ�
    cudaFree(d_in);
    cudaFree(d_out);

    return 0;
}
