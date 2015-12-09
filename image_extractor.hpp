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
        ImageExtractor(Mat image);
        vector<ImageData> extract();
        void showImage(Mat image);
    
    private:
        Mat image, grayImage, binImage, output;

        void binarize(int threshold=180);
        vector<Points> getContours();
        vector<Points> fillObject(Mat image, vector<Point> contour);
        int getColor(Points obj);
        Vec3b getBGR(int color);
        Point getCentroid(Points contour);
        double getOrientation(Points contour);
        void drawDot(Mat image, Point pt);
        void drawLine(Mat image, double theta, Point cen);
        void drawText(Mat image, string text, Point pt);
};

#endif
