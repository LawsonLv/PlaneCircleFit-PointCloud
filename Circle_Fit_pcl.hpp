
#include<pcl/segmentation/sac_segmentation.h>
#include<pcl/features/boundary.h>

using namespace::std;

template<typename PointT>
class CircleFit
{
public:
	typedef pcl::PointCloud<PointT> cloudType;
	typedef typename cloudType::Ptr pCloudType;
	pCloudType m_pCloudIn;
	pCloudType m_pCloudBoundary;
	pcl::ModelCoefficients::Ptr m_pCoefPlane;

public:
	CircleFit();
	~CircleFit();

	inline void
		setCloudInput(pCloudType pCloudIn)
	{
			m_pCloudIn = pCloudIn;
		}
	Eigen::Vector3f GetCentroid();
	Eigen::Vector3f GetLeastSquareCenter();
	Eigen::Vector3f GetThreePointCenter();

private:
	void computePlane();
	void ThreeDtoTwoD();
	void computeBoundary();
	void LeastSquareCircleFit();
	void ThreePointCircleFit();

	Eigen::Vector3f m_circle_center;
	Eigen::Matrix3f m_transMat;
};

template<typename PointT>
CircleFit<PointT>::CircleFit()
{
	pCloudType pCloudOut(new cloudType);
	m_pCloudBoundary = pCloudOut;
	pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
	m_pCoefPlane = coefficients;
}

template<typename PointT>
CircleFit<PointT>::~CircleFit()
{

}

template<typename PointT>
Eigen::Vector3f CircleFit<PointT>::GetCentroid()
{
	Eigen::Vector4f centroid4f;
	pcl::compute3DCentroid(*m_pCloudIn, centroid4f);
	Eigen::Vector3f centroid(centroid4f[0], centroid4f[1], centroid4f[2]);
	m_circle_center = centroid;
	return centroid;
}

template<typename PointT>
Eigen::Vector3f CircleFit<PointT>::GetLeastSquareCenter()
{
	computePlane();
	ThreeDtoTwoD();
	computeBoundary();
	LeastSquareCircleFit();
	return m_circle_center;
}

template<typename PointT>
Eigen::Vector3f CircleFit<PointT>::GetThreePointCenter()
{
	computePlane();
	ThreeDtoTwoD();
	computeBoundary();
	ThreePointCircleFit();
	return m_circle_center;
}

template<typename PointT>
void CircleFit<PointT>::computePlane()
{
	pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
	pcl::SACSegmentation<PointT> seg;
	seg.setOptimizeCoefficients(true);
	seg.setModelType(pcl::SACMODEL_PLANE);
	seg.setMethodType(pcl::SAC_RANSAC);
	seg.setDistanceThreshold(5); // 5mm平面距离
	seg.setInputCloud(m_pCloudIn);
	seg.segment(*inliers, *m_pCoefPlane);
}

template<typename PointT>
void CircleFit<PointT>::ThreeDtoTwoD()//Livox 右手系
{
	int i = rand() % (m_pCloudIn->size() - 1);
	int j = rand() % (m_pCloudIn->size() - 1);
	if (j == i)
		j++;
	Eigen::Vector3f center(m_pCloudIn->points[i].x, m_pCloudIn->points[i].y, m_pCloudIn->points[i].z);
	Eigen::Vector3f y_point(m_pCloudIn->points[j].x, m_pCloudIn->points[j].y, m_pCloudIn->points[j].z);
	//cout << center[0] << " " << center[1] << " " << center[2] << endl;
	//cout << y_point[0] << " " << y_point[1] << " " << y_point[2] << endl;
	Eigen::Vector3f plane_normal(m_pCoefPlane->values[0], m_pCoefPlane->values[1], m_pCoefPlane->values[2]);
	if (plane_normal[0] >= plane_normal[1] && plane_normal[0] >= plane_normal[2])
	{
		center[0] = -(center[1] * plane_normal[1] + center[2] * plane_normal[2] + m_pCoefPlane->values[3]) / plane_normal[0]; //保证圆心在平面上
		y_point[0] = -(y_point[1] * plane_normal[1] + y_point[2] * plane_normal[2] + m_pCoefPlane->values[3]) / plane_normal[0];
	}
	else if (plane_normal[1] >= plane_normal[0] && plane_normal[1] >= plane_normal[2])
	{
		center[1] = -(center[0] * plane_normal[0] + center[2] * plane_normal[2] + m_pCoefPlane->values[3]) / plane_normal[1];
		y_point[1] = -(y_point[0] * plane_normal[0] + y_point[2] * plane_normal[2] + m_pCoefPlane->values[3]) / plane_normal[1];
	}
	else if (plane_normal[2] >= plane_normal[0] && plane_normal[2] >= plane_normal[1])
	{
		center[2] = -(center[0] * plane_normal[0] + center[1] * plane_normal[1] + m_pCoefPlane->values[3]) / plane_normal[2];
		y_point[2] = -(y_point[0] * plane_normal[0] + y_point[1] * plane_normal[1] + m_pCoefPlane->values[3]) / plane_normal[2];
	}
	//cout << center[0] << " " << center[1] << " " << center[2] << endl;
	//cout << y_point[0] << " " << y_point[1] << " " << y_point[2] << endl;
	Eigen::Vector3f y_normal = y_point - center;
	y_normal.normalize();
	Eigen::Vector3f z_normal = plane_normal.cross(y_normal);
	z_normal.normalize();
	Eigen::Matrix3f tranMat;
	tranMat.col(0) = Eigen::Vector3f(plane_normal[0], plane_normal[1], plane_normal[2]);
	tranMat.col(1) = Eigen::Vector3f(y_normal[0], y_normal[1], y_normal[2]);
	tranMat.col(2) = Eigen::Vector3f(z_normal[0], z_normal[1], z_normal[2]);
	cout << "TranMat: " << tranMat << endl;
	m_circle_center = center;
	m_transMat = tranMat;
}

template<typename PointT>
void CircleFit<PointT>::computeBoundary()
{
	pcl::PointCloud<pcl::Normal>::Ptr Normals(new pcl::PointCloud<pcl::Normal>);
	pcl::PointCloud<pcl::Boundary> boundaries;
	pcl::BoundaryEstimation<PointT, pcl::Normal, pcl::Boundary> Est;
	pcl::search::KdTree<PointT>::Ptr tree(new pcl::search::KdTree<PointT>);

	pcl::NormalEstimation<PointT, pcl::Normal> normEst;
	normEst.setInputCloud(m_pCloudIn);
	normEst.setSearchMethod(tree);
	normEst.setKSearch(20);
	normEst.compute(*Normals);

	Est.setInputCloud(m_pCloudIn);
	Est.setInputNormals(Normals);
	Est.setSearchMethod(tree);
	Est.setKSearch(10);
	Est.compute(boundaries);

	for (int i = 0; i < m_pCloudIn->points.size(); i++)
	{
		uint8_t x = boundaries.points[i].boundary_point;
		int Boundflag = static_cast<int>(x);
		if (Boundflag == 1)
			m_pCloudBoundary->push_back(m_pCloudIn->points[i]);
	}
}

template<typename PointT>
void CircleFit<PointT>::LeastSquareCircleFit()
{
	vector<Eigen::Vector3f> circle_point;
	for (int i = 0; i < m_pCloudBoundary->size(); i++)
	{
		Eigen::Vector3f point(m_pCloudBoundary->points[i].x, m_pCloudBoundary->points[i].y, m_pCloudBoundary->points[i].z);
		point = point - m_circle_center;
		point = m_transMat.inverse()*point;
		//cout << point[0] << " " << point[1] << " " << point[2] << endl;
		circle_point.push_back(point);
	}
	int pointNum = circle_point.size();
	if (pointNum < 3)
		return;

	double X1 = 0.0;
	double Y1 = 0.0;
	double X2 = 0.0;
	double Y2 = 0.0;
	double X3 = 0.0;
	double Y3 = 0.0;
	double X1Y1 = 0.0;
	double X1Y2 = 0.0;
	double X2Y1 = 0.0;
	for (int i = 0; i < circle_point.size(); i++)
	{
		X1 = X1 + circle_point[i][1];
		Y1 = Y1 + circle_point[i][2];
		X2 = X2 + circle_point[i][1] * circle_point[i][1];
		Y2 = Y2 + circle_point[i][2] * circle_point[i][2];
		X3 = X3 + circle_point[i][1] * circle_point[i][1] * circle_point[i][1];
		Y3 = Y3 + circle_point[i][2] * circle_point[i][2] * circle_point[i][2];
		X1Y1 = X1Y1 + circle_point[i][1] * circle_point[i][2];
		X1Y2 = X1Y2 + circle_point[i][1] * circle_point[i][2] * circle_point[i][2];
		X2Y1 = X2Y1 + circle_point[i][1] * circle_point[i][1] * circle_point[i][2];
	}
	double C = pointNum*X2 - X1*X1;
	double D = pointNum*X1Y1 - X1*Y1;
	double E = pointNum*X3 + pointNum*X1Y2 - (X2 + Y2)*X1;
	double G = pointNum*Y2 - Y1*Y1;
	double H = pointNum*X2Y1 + pointNum*Y3 - (X2 + Y2)*Y1;

	double a = (H*D - E*G) / (C*G - D*D);
	double b = (H*C - E*D) / (D*D - C*G);
	double c = -(a*X1 + b*Y1 + X2 + Y2) / pointNum;

	Eigen::Vector3f circle_center(0, -a / 2, -b / 2);
	m_circle_center = m_transMat*circle_center + m_circle_center;
}

template<typename PointT>
void CircleFit<PointT>::ThreePointCircleFit()//前提条件是必须超过半圆
{
	vector<Eigen::Vector3f> circle_point;
	for (int i = 0; i < m_pCloudBoundary->size(); i++)
	{
		Eigen::Vector3f point(m_pCloudBoundary->points[i].x, m_pCloudBoundary->points[i].y, m_pCloudBoundary->points[i].z);
		point = point - m_circle_center;
		point = m_transMat.inverse()*point;
		//cout << point[0] << " " << point[1] << " " << point[2] << endl;
		circle_point.push_back(point);
	}
	int pointNum = circle_point.size();
	if (pointNum < 3)
		return;
	int indexA, indexB, indexC;
	Eigen::Vector3f pointA, pointB, pointC;
	double disMax = 0.0;
	for (int i = 0; i < circle_point.size(); i++)
	{
		pointA = circle_point[i];
		for (int j = i + 1; j < circle_point.size(); j++)
		{
			pointB = circle_point[j];
			double distance = (pointA - pointB).dot(pointA - pointB);
			if (distance > disMax)
			{
				disMax = distance;
				indexA = i;
				indexB = j;
			}
		}
	}
	pointA = circle_point[indexA];
	pointB = circle_point[indexB];
	circle_center = (pointA + pointB) / 2;
	circle_center[0] = 0.0;
	disMax = 0.0;
	for (int i = 0; i < circle_point.size(); i++)
	{
		pointC = circle_point[i];
		double distance = (m_circle_center - pointC).dot(m_circle_center - pointC);
		if (distance > disMax)
		{
			disMax = distance;
			indexC = i;
		}
	}
	if (indexC != indexA && indexC != indexB)
	{
		double a = 2 * (circle_point[indexB][1] - circle_point[indexA][1]);
		double b = 2 * (circle_point[indexB][2] - circle_point[indexA][2]);
		double c = circle_point[indexB][2] * circle_point[indexB][2] + circle_point[indexB][1] * circle_point[indexB][1] - circle_point[indexA][2] * circle_point[indexA][2] - circle_point[indexA][1] * circle_point[indexA][1];
		double d = 2 * (circle_point[indexC][1] - circle_point[indexB][1]);
		double e = 2 * (circle_point[indexC][2] - circle_point[indexB][2]);
		double f = circle_point[indexC][2] * circle_point[indexC][2] + circle_point[indexC][1] * circle_point[indexC][1] - circle_point[indexB][2] * circle_point[indexB][2] - circle_point[indexB][1] * circle_point[indexB][1];
		circle_center[1] = (b*f - e*c) / (b*d - e*a);
		circle_center[2] = (d*c - a*f) / (b*d - e*a);
	}

	m_circle_center = m_transMat*circle_center + m_circle_center;
}