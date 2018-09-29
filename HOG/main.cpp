#include <iostream>
#include <opencv2/core/core.hpp>  
#include <opencv2/highgui/highgui.hpp>  
#include "fstream"
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

#define FALSE 0;
#define TRUE 1;


bool creatTrainSample() {
	string FileName;
	ifstream filein("D:\\data\\seg\\trainsamples.txt");
	if (!filein)
		return FALSE;
	ofstream out;
	out.open("D:\\data\\tran.txt", ios::trunc); //ios::trunc表示在打开文件前将文件清空,由于是写入,文件不存在则创建
	while (getline(filein, FileName))//逐行读取数据并存于s中，直至数据全部读取
	{
		out << "D:\\data\\dot\\" << FileName << '\n';//路径后面加上训练样本的filename
		int n = FileName[0] - '0';//每个训练样本文件都以数字开头命令，数字即代表该文件的类别
		/*if (n > 9)
			n = FileName[0];*/
		out << n << '\n';//每个样本的后面写入其类别，用于SVM训练时指定type
	}
	filein.close();
	out.close();
	return TRUE;
}
bool mySVM_train() {
	vector<string> img_path;//图像路径容器  
	vector<int> img_catg;//图像类别容器
	int nLine = 0;
	string buf;
	ifstream svm_data("D:\\data\\dot\\train.txt");//训练样本图片的路径都写在这个txt文件中，使用bat批处理文件可以得到这个txt文件 
	if (!svm_data)
		return FALSE;
	unsigned long n;
	while (svm_data)//将训练样本文件依次读取进来    
	{
		if (getline(svm_data, buf))
		{
			nLine++;
			if (nLine % 2 == 0)//注：奇数行是图片全路径，偶数行是标签 
			{
				img_catg.push_back(atoi(buf.c_str()));//atoi将字符串转换成整型，标志(0,1，2，...，9)，注意这里至少要有两个类别，否则会出错    
			}
			else
			{
				img_path.push_back(buf);//图像路径    
			}
		}
	}
	svm_data.close();//关闭文件    
	CvMat *data_mat, *res_mat;
	int nImgNum = nLine / 2; //nImgNum是样本数量，只有文本行数的一半，另一半是标签     
	data_mat = cvCreateMat(nImgNum, 225, CV_32FC1);  //第二个参数，即矩阵的列是由下面的descriptors的大小决定的，可以由descriptors.size()得到，且对于不同大小的输入训练图片，这个值是不同的  
	cvSetZero(data_mat);
	//类型矩阵,存储每个样本的类型标志    
	res_mat = cvCreateMat(nImgNum, 1, CV_32FC1);
	cvSetZero(res_mat);
	IplImage* src;
	IplImage* trainImg = cvCreateImage(cvSize(20, 35), 8, 3);//需要分析的图片，这里车标的尺寸归一化至40*32，所以上面定义了432，如果要更改图片大小，可以先用debug查看一下descriptors是多少，然后设定好再运行    

														   //处理HOG特征  
	for (string::size_type i = 0; i != img_path.size(); i++)
	{
		src = cvLoadImage(img_path[i].c_str(), 1);
		if (src == NULL)
		{
			cout << " Can not load the image: " << img_path[i].c_str() << endl;
			continue;
		}

		cout << " 处理： " << img_path[i].c_str() << endl;

		cvResize(src, trainImg);
		HOGDescriptor *hog = new HOGDescriptor(cvSize(20, 35), cvSize(4, 7), cvSize(4, 7), cvSize(4, 7), 9);//图片尺寸：40*32；block尺寸：16*16；cell尺寸：8*8；检测窗口的滑动步长：8*8；一个单元格内统计9个方向的梯度直方图
		vector<float>descriptors;//存放结果     
		hog->compute(trainImg, descriptors, Size(1, 1), Size(0, 0)); //Hog特征计算      
		cout << "HOG dims: " << descriptors.size() << endl;
		n = 0;
		for (vector<float>::iterator iter = descriptors.begin(); iter != descriptors.end(); iter++)
		{
			cvmSet(data_mat, i, n, *iter);//存储HOG特征 
			n++;
		}
		cvmSet(res_mat, i, 0, img_catg[i]);
		cout << " 处理完毕: " << img_path[i].c_str() << " " << img_catg[i] << endl;
	}


	//    CvSVM svm = CvSVM();//新建一个SVM     
	CvSVM svm;
	CvSVMParams param;//这里是SVM训练相关参数  
	CvTermCriteria criteria;
	criteria = cvTermCriteria(CV_TERMCRIT_EPS, 1000, FLT_EPSILON);
	param = CvSVMParams(CvSVM::C_SVC, CvSVM::RBF, 10.0, 0.09, 1.0, 10.0, 0.5, 1.0, NULL, criteria);
	//        param = CvSVMParams(CvSVM::C_SVC, CvSVM::RBF, 10.0, 0.3, 1.0, 5, 0.5, 1.0, NULL, criteria);
	svm.train(data_mat, res_mat, NULL, NULL, param);//训练数据     
													//保存训练好的分类器      
	svm.save("D:\\data\\MySVM.xml");
	cvReleaseMat(&data_mat);
	cvReleaseMat(&res_mat);
	cvReleaseImage(&trainImg);
	return TRUE;
}
int main() {
	creatTrainSample();
	return 0;
}