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
	out.open("D:\\data\\tran.txt", ios::trunc); //ios::trunc��ʾ�ڴ��ļ�ǰ���ļ����,������д��,�ļ��������򴴽�
	while (getline(filein, FileName))//���ж�ȡ���ݲ�����s�У�ֱ������ȫ����ȡ
	{
		out << "D:\\data\\dot\\" << FileName << '\n';//·���������ѵ��������filename
		int n = FileName[0] - '0';//ÿ��ѵ�������ļ��������ֿ�ͷ������ּ�������ļ������
		/*if (n > 9)
			n = FileName[0];*/
		out << n << '\n';//ÿ�������ĺ���д�����������SVMѵ��ʱָ��type
	}
	filein.close();
	out.close();
	return TRUE;
}
bool mySVM_train() {
	vector<string> img_path;//ͼ��·������  
	vector<int> img_catg;//ͼ���������
	int nLine = 0;
	string buf;
	ifstream svm_data("D:\\data\\dot\\train.txt");//ѵ������ͼƬ��·����д�����txt�ļ��У�ʹ��bat�������ļ����Եõ����txt�ļ� 
	if (!svm_data)
		return FALSE;
	unsigned long n;
	while (svm_data)//��ѵ�������ļ����ζ�ȡ����    
	{
		if (getline(svm_data, buf))
		{
			nLine++;
			if (nLine % 2 == 0)//ע����������ͼƬȫ·����ż�����Ǳ�ǩ 
			{
				img_catg.push_back(atoi(buf.c_str()));//atoi���ַ���ת�������ͣ���־(0,1��2��...��9)��ע����������Ҫ��������𣬷�������    
			}
			else
			{
				img_path.push_back(buf);//ͼ��·��    
			}
		}
	}
	svm_data.close();//�ر��ļ�    
	CvMat *data_mat, *res_mat;
	int nImgNum = nLine / 2; //nImgNum������������ֻ���ı�������һ�룬��һ���Ǳ�ǩ     
	data_mat = cvCreateMat(nImgNum, 225, CV_32FC1);  //�ڶ���������������������������descriptors�Ĵ�С�����ģ�������descriptors.size()�õ����Ҷ��ڲ�ͬ��С������ѵ��ͼƬ�����ֵ�ǲ�ͬ��  
	cvSetZero(data_mat);
	//���;���,�洢ÿ�����������ͱ�־    
	res_mat = cvCreateMat(nImgNum, 1, CV_32FC1);
	cvSetZero(res_mat);
	IplImage* src;
	IplImage* trainImg = cvCreateImage(cvSize(20, 35), 8, 3);//��Ҫ������ͼƬ�����ﳵ��ĳߴ��һ����40*32���������涨����432�����Ҫ����ͼƬ��С����������debug�鿴һ��descriptors�Ƕ��٣�Ȼ���趨��������    

														   //����HOG����  
	for (string::size_type i = 0; i != img_path.size(); i++)
	{
		src = cvLoadImage(img_path[i].c_str(), 1);
		if (src == NULL)
		{
			cout << " Can not load the image: " << img_path[i].c_str() << endl;
			continue;
		}

		cout << " ���� " << img_path[i].c_str() << endl;

		cvResize(src, trainImg);
		HOGDescriptor *hog = new HOGDescriptor(cvSize(20, 35), cvSize(4, 7), cvSize(4, 7), cvSize(4, 7), 9);//ͼƬ�ߴ磺40*32��block�ߴ磺16*16��cell�ߴ磺8*8����ⴰ�ڵĻ���������8*8��һ����Ԫ����ͳ��9��������ݶ�ֱ��ͼ
		vector<float>descriptors;//��Ž��     
		hog->compute(trainImg, descriptors, Size(1, 1), Size(0, 0)); //Hog��������      
		cout << "HOG dims: " << descriptors.size() << endl;
		n = 0;
		for (vector<float>::iterator iter = descriptors.begin(); iter != descriptors.end(); iter++)
		{
			cvmSet(data_mat, i, n, *iter);//�洢HOG���� 
			n++;
		}
		cvmSet(res_mat, i, 0, img_catg[i]);
		cout << " �������: " << img_path[i].c_str() << " " << img_catg[i] << endl;
	}


	//    CvSVM svm = CvSVM();//�½�һ��SVM     
	CvSVM svm;
	CvSVMParams param;//������SVMѵ����ز���  
	CvTermCriteria criteria;
	criteria = cvTermCriteria(CV_TERMCRIT_EPS, 1000, FLT_EPSILON);
	param = CvSVMParams(CvSVM::C_SVC, CvSVM::RBF, 10.0, 0.09, 1.0, 10.0, 0.5, 1.0, NULL, criteria);
	//        param = CvSVMParams(CvSVM::C_SVC, CvSVM::RBF, 10.0, 0.3, 1.0, 5, 0.5, 1.0, NULL, criteria);
	svm.train(data_mat, res_mat, NULL, NULL, param);//ѵ������     
													//����ѵ���õķ�����      
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