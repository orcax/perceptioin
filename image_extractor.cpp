#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <cstring>
#include <stdio.h>
#include "image_extractor.hpp"

using namespace cv;
using namespace std;

/**********************************************/
/*************** Public Methods ***************/
/**********************************************/

ImageExtractor::ImageExtractor(Mat image)
{
    this->image = image;
    this->output = Mat::zeros(image.size(), CV_8UC3); 
    this->bgr = Scalar(255, 255, 255);
}

vector<ImageData> ImageExtractor::extract()
{
    vector<ImageData> imageData;
    vector<Points> contours = this->getContours();
    for(int i=0;i<contours.size();i++)
    {
        ImageData id;
        id.centroid = this->getCentroid(contours[i]);
        id.orient = this->getOrientation(contours[i]);
        id.color = this->getColor(contours[i]);

        imageData.push_back(id);
        //cout << obj.size() << endl;
        //this->plotPoints(obj);
        //this->showImage(this->output, "ccc");
    }
    return imageData;
}

void ImageExtractor::setOutputColor(uchar b, uchar g, uchar r)
{
    this->bgr = Scalar(b, g, r);
}

void ImageExtractor::setOutputColor(char color)
{
    switch(color)
    {
        case R:
            this->setOutputColor(0, 0, 250);
            break;
        case G:
            this->setOutputColor(0, 250, 0);
            break;
        case B:
            this->setOutputColor(250, 0, 0);
            break;
        case Y:
            this->setOutputColor(0, 250, 250); 
            break;
        default:
            this->setOutputColor(255, 255, 255);
    }
}

void ImageExtractor::plotDot(Point pt)
{
    this->plotDot(pt, this->output);
}

void ImageExtractor::plotDot(Point pt, Mat image)
{
    circle(image, pt, 4, this->bgr);
}

void ImageExtractor::plotPoints(Points pts)
{
    this->plotPoints(pts, this->output);
}

void ImageExtractor::plotPoints(Points pts, Mat image)
{
    for(int i=0;i<pts.size();i++)
    {
        circle(image, pts[i], 1, this->bgr);
    }
}

void ImageExtractor::plotLine(double theta, Point start)
{
    this->plotLine(theta, start, this->output);
}

void ImageExtractor::plotLine(double theta, Point start, Mat image)
{
    Point end = Point(start.x + 50 * cos(theta), start.y + 50 * sin(theta));
    line(this->output, start, end, this->bgr, 3);
}

void ImageExtractor::plotText(string text, Point pt) 
{
    this->plotText(text, pt, this->output);
}

void ImageExtractor::plotText(string text, Point pt, Mat image) 
{
    putText(image, text, pt, FONT_HERSHEY_SIMPLEX, 1, this->bgr, 2);
}

void ImageExtractor::showImage(Mat image, string title)
{
    namedWindow(title, CV_WINDOW_NORMAL);
    imshow(title, image);
    waitKey(0);
}


/**********************************************/
/************** Private Methods ***************/
/**********************************************/

vector<Points> ImageExtractor::getContours()
{
    cvtColor(this->image, this->grayImage, CV_BGR2GRAY);
    //GaussianBlur(this->grayImage, this->grayImage, Size(9, 9), 2, 2);
    //Canny(this->grayImage, this->binImage, 0, 50, 5);
    threshold(this->grayImage, this->binImage, 160, 255, 0);
    Mat dstImage;
    //vector<Vec3f> circles;
    //HoughCircles(this->grayImage, circles, CV_HOUGH_GRADIENT, 1, 10, 200, 100, 0, 0);
    //cout << circles.size() << endl;
    //for(int i=0;i<circles.size();i++)
    //{
    //    Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
    //    int radius = cvRound(circles[i][2]);
    //    circle(output, center, radius, Scalar(255,255,255), 3, 8, 0);
    //}

    vector<Points> contours, result;
    vector<Vec4i> hierarchy;
    Mat element = getStructuringElement(MORPH_ELLIPSE, Size(15, 15));
    morphologyEx(this->binImage, dstImage, MORPH_GRADIENT, element);
    findContours(dstImage, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0,0));
    //findContours(this->binImage, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
    Points approx;
    this->setOutputColor(R);
    for(int i=0;i<contours.size();i++)
    {
        Points obj = fillObject(contours[i]);
        if(obj.size() < 1000 || obj.size() > 20000) continue;
        //approxPolyDP(Mat(contours[i]), approx, arcLength(Mat(contours[i]), true) * 0.02, true);
        //for(int j=0;j<approx.size();j++)
        //{
        //    this->plotDot(approx[j]);
        //    cout << approx[j];
        //}
        //cout << endl << "-----contour " << approx.size() << endl;
        result.push_back(contours[i]);
    }
    this->setOutputColor(W);
    return result;
}

Points ImageExtractor::fillObject(Points contour)
{
    Mat tmp = Mat::zeros(this->image.size(), CV_8UC3);
    vector<Points> contours;
    contours.push_back(contour);
    const int mask = 1;
    drawContours(tmp, contours, 0, Scalar(mask, mask, mask), CV_FILLED);
    Points obj;
    for(int x=0;x<tmp.cols;x++)
    {
        for(int y=0;y<tmp.rows;y++)
        {
            Vec3b c = tmp.at<Vec3b>(y, x);
            if(c[0] == mask)
            {
                obj.push_back(Point(x, y));
            }
        }
    }
    return obj;
}

char ImageExtractor::getColor(Points obj)
{
    Vec3b bgr = this->image.at<Vec3b>(obj[0].y, obj[0].x);
    long b = (long)bgr[0], g = (long)bgr[1], r = (long)bgr[2];
    const int K = 250;
    int count = 1;
    for(int i=1;i<obj.size();i++)
    {
        Vec3b tmp = image.at<Vec3b>(obj[i].y, obj[i].x);
        if(tmp[0] > K && tmp[1] > K && tmp[2] > K)
        {
            continue;
        }
        b += (long)tmp[0];
        g += (long)tmp[1];
        r += (long)tmp[2];
        count++;
    }
    b /= count;
    g /= count;
    r /= count;
    //cout << b << " " << g << " "<< r << " ";
    const int M = 5;
    if(b>r+M && b>g+M && b > 200) return B;
    if(g>b+M && g>r+M && g > 200) return G;
    if(r>b+M && r>g+M && r > 198) return R;
    return Y;
}

Vec3b ImageExtractor::getBGR(int color)
{
    Vec3b brg(0, 0, 0);
    switch(color)
    {
        case R: 
            brg[2] = 255; break;
        case G: 
            brg[1] = 255; break;
        case B: 
            brg[0] = 255; break;
        case Y: 
            brg[1] = 205; brg[2] = 255; break;
    }
    return brg;
}

Point ImageExtractor::getCentroid(Points contour)
{
    Moments mu = moments(contour, false);
    Point mc = Point(mu.m10/mu.m00, mu.m01/mu.m00);
    return mc;
}

double ImageExtractor::getOrientation(Points contour)
{
    Moments mo = moments(contour,false);
    double tan_v = (2 * mo.mu11) / (mo.mu20 - mo.mu02);
    double dtheta = atan(tan_v) / 2;
    if((mo.mu20 - mo.mu02 ) < 0)
    {
        dtheta += M_PI / 2;
    }
    return dtheta;
    //double rate = tan(dtheta);
    //cout << "The rate is "<<rate<<endl;
    //return rate;
}

