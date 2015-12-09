#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <cstring>
#include <stdio.h>
#include "image_extractor.hpp"

using namespace cv;
using namespace std;

ImageExtractor::ImageExtractor(Mat image)
{
    this.image = image;
    this.binarize();
}

vector<ImageData> extract()
{
    //TODO
    return NULL;
}

void ImageExtractor::binarize(int threshold)
{
    cvtColor(this.image, this.grayImage, CV_BGR2GRAY);
    threshold(this.grayImage, this.binImage, threshold, 255, 0);
}

vector<Points> ImageExtractor::getContours()
{
    Mat dstImage;
    vector<Contour> contours;
    vector<Vec4i> hierarchy;

    //Mat dist;
    //distanceTransform(image_src, dist, CV_DIST_L2, 3);
    //normalize(dist, dist, 0, 1.0, NORM_MINMAX);
    //threshold(dist, dist, 0.65, 1.0, CV_THRESH_BINARY);
    //Mat dist_8u;
    //dist.convertTo(dist_8u, CV_8U);
    //findContours(dist_8u, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);

    Mat element = getStructuringElement(MORPH_RECT, Size(10, 10));
    morphologyEx(this.binImage, dstImage, MORPH_OPEN, element);
    findContours(dstImage, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0,0));
    return contours;
}

Points ImageExtractor::fillObject(Points contour)
{
    Mat tmp = Mat::zeros(this.image.size(), CV_8UC3);
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

int ImageExtractor::getColor(Points obj)
{
    Vec3b bgr = this.image.at<Vec3b>(obj[0].y, obj[0].x);
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
    if(b>r+M && b>g+M && b > 200) return BLUE;
    if(g>b+M && g>r+M && g > 200) return GREEN;
    if(r>b+M && r>g+M && r > 198) return RED;
    return YELLOW;
}

Vec3b ImageExtractor::getBGR(int color)
{
    Vec3b brg(0, 0, 0);
    switch(color)
    {
        case BLUE: 
            brg[0] = 255; break;
        case GREEN: 
            brg[1] = 255; break;
        case RED: 
            brg[2] = 255; break;
        case YELLOW: 
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

void ImageExtractor::drawDot(Point pt)
{
    circle(this.output, pt, 4, Scalar(255,255,255), -1, 8, 0);
}

void ImageExtractor::drawLine(double theta, Point cen)
{
    Point end = Point(cen.x + 50 * cos(theta), cen.y + 50 * sin(theta));
    //line(img,cen,end,Scalar(0, 0, 0), 3);
    line(this.output,cen,end,Scalar(255, 255, 255), 3);
}

void ImageExtractor::drawText(string text, Point pt) 
{
    Scalar color(255, 255, 255);
    putText(this.output, text, pt, FONT_HERSHEY_SIMPLEX, 1, color, 2);
}

void ImageExtractor::showImage(Mat image)
{
    namedWindow("Display", CV_WINDOW_NORMAL);
    imshow("Display", image);
    waitKey(0);
}

