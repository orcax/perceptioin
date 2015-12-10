#pragma once
#ifndef IMAGE_EXTRACTOR_HPP
#define IMAGE_EXTRACTOR_HPP

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <cstring>
#include <stdio.h>

using namespace cv;
using namespace std;

#define R 'r'
#define G 'g'
#define B 'b'
#define Y 'y'
#define W 'w'

typedef struct image_data_ 
{
    Point centroid;
    double orient;
    char color;
    bool stand;
} ImageData;

typedef vector<Point> Points;

class ImageExtractor
{
    public:
        Mat image, grayImage, binImage, output;

        ImageExtractor(Mat image);
        vector<ImageData> extract();
        void setOutputColor(uchar b, uchar g, uchar r);
        void setOutputColor(char color);
        void plotDot(Point pt);
        void plotDot(Point pt, Mat image);
        void plotPoints(Points pts);
        void plotPoints(Points pts, Mat image);
        void plotLine(double theta, Point start);
        void plotLine(double theta, Point start, Mat image);
        void plotText(string text, Point pt);
        void plotText(string text, Point pt, Mat image);
        void showImage(Mat image, string title="Display");
    
    private:
        Scalar bgr;

        vector<Points> getContours();
        Points fillObject(Points contour);
        char getColor(Points obj);
        Vec3b getBGR(int color);
        Point getCentroid(Points contour);
        double getOrientation(Points contour);
};

#endif
