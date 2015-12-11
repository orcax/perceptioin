#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <cstring>
#include <stdio.h>
#include "image_extractor.hpp"

using namespace cv;
using namespace std;


int main(int argc, char** argv)
{
    Mat image = imread(argv[1], CV_LOAD_IMAGE_COLOR);
    if(!image.data) 
    {
        cout << "Could not open image " << argv[1] << endl;
        return -1;
    }

    ImageExtractor ie(image);
    vector<ImageData> imagedata = ie.extract();

    //ie.showImage(ie.binImage, "aaa");
    //ie.showImage(ie.output, "ccc");
    /*
    Mat image_binary = ie.binarize(image);
    Mat output = Mat::zeros(image_binary.size(), CV_8UC3);
    vector<vector<Point> > contours = get_contours(image_binary);
    vector<vector<Point> > objects;
    cout << argv[1] << endl;
    for(int i=0;i<contours.size();i++)
    {
        vector<Point> contour = contours[i];
        vector<Point> obj = fill_object(image, contour);
        objects.push_back(obj);
        //cout << obj.size() << endl;
        int color = get_color(image, contour);
        //cout << "color=" << color << "; ";
        Vec3b brg = get_brg(color);
        drawContours(output, contours, i, Scalar(brg[0], brg[1], brg[2]), CV_FILLED);
        for(int j=0;j<obj.size();j++)
        {
            int x = obj[j].x;
            int y = obj[j].y;
            output.at<Vec3b>(y, x)[0] = brg[0];
            output.at<Vec3b>(y, x)[1] = brg[1];
            output.at<Vec3b>(y, x)[2] = brg[2];
        }

        Point center = centroid(contour);
        drawDot(output, center);

        double theta = getOrientation(contour); 
        cout << center << " " << theta << endl;
        drawLine(output, theta, center);

        char text[1000];
        sprintf(text, "(%d,%d,%lf)", center.x, center.y, theta);
        drawText(output, text, center);
    }
    char text[100];
    sprintf(text, "Number of objects: %ld", contours.size());
    drawText(output, text, Point(0,200));
    */

    //imwrite(argv[2], output);

    return 0;
}
