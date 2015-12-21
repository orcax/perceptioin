#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <algorithm>
#include <cstring>
#include <math.h>
#include <stdio.h>
#include "image_extractor.hpp"

using namespace cv;
using namespace std;

/**********************************************/
/*************** Public Methods ***************/
/**********************************************/

ImageExtractor::ImageExtractor(Mat image)
{
    resize(image, this->image, Size(WIDTH, HEIGHT));
    //this->output = Mat::zeros(this->image.size(), CV_8UC3); 
    this->output = this->image.clone();
    this->bgr = Scalar(255, 255, 255);
}

void ImageExtractor::extract(vector<ImageObject>& imageObjects)
{
    this->preprocess();
    vector<Points> contours = this->getContours();
    char colors[9];
    this->getColors(contours, colors);
    for(int i=0;i<contours.size();i++)
    {
        ImageObject io;
        io.centroid = this->getCentroid(contours[i]);
        io.orient = this->getOrientation(contours[i]);
        io.color = colors[i];
        io.stand = this->isStand(contours[i], io.centroid);
        if(io.stand) io.orient = 0.0;
        imageObjects.push_back(io);

        Points obj = this->fillObject(contours[i]);
        this->setOutputColor(io.color);
        this->plotPoints(obj);
        if(!io.stand) 
        {
            this->setOutputColor('w');
            this->plotLine(io.orient, io.centroid);
        }
        //this->showImage(this->output, "ccc");
    }
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
        case W:
            this->setOutputColor(255, 255, 255); 
            break;
        default:
            this->setOutputColor(0, 0, 0);
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
    int len = 40;
    Point end = Point(start.x + len * cos(theta), start.y + len * sin(theta));
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

void ImageExtractor::preprocess()
{
    const int thresh1 = 180;
    Mat image2 = this->image.clone();
    /*
    for(int i=0;i<HEIGHT;i++)
    {
        for(int j=0;j<WIDTH;j++)
        {
            Vec3b* color = &image2.at<Vec3b>(i,j);
            int b = (*color)[0];
            int g = (*color)[1];
            int r = (*color)[2];
            (*color)[0] = b > thresh1 ? 255 : 0;
            (*color)[1] = g > thresh1 ? 255 : 0;
            (*color)[2] = r > thresh1 ? 255 : 0;
            //if(b > g && b > r && b > thresh1) (*color)[0] = 255;
            //else if(g > b && g > r && g > thresh1) (*color)[0] = 255;
            //else if(r > b && r > g && r > thresh1) (*color)[0] = 255;
        }
    }
    */

    const int thresh2 = 170;
    const int erosion_type = MORPH_ELLIPSE;

    Mat grayImage, erodeImage;

    //TODO change erosion
    cvtColor(image2, grayImage, CV_BGR2GRAY);
    //GaussianBlur(this->grayImage, this->grayImage, Size(9, 9), 2, 2);
    //Canny(this->grayImage, this->binImage, 0, 50, 5); 
    threshold(grayImage, erodeImage, thresh2, 255, 0);
    Mat element = getStructuringElement(erosion_type, Size(10, 10));
    //morphologyEx(binImage, erodeImage, MORPH_OPEN, element);
    for(int i=0;i<3;i++) {
        erode(erodeImage, erodeImage, element);
        dilate(erodeImage, erodeImage, element);
    }

    Mat labelImage(this->image.size(), CV_32S);
    int nLabels = this->connectedComponents(labelImage, erodeImage, 8);
    /*
    Vec3b colors[nLabels];
    colors[0] = Vec3b(0, 0, 0);//background
    for(int label = 1; label < nLabels; ++label){
        colors[label] = Vec3b( (rand()&255), (rand()&255), (rand()&255) );
    }
    */
    Mat dstImage(this->image.size(), CV_8UC3);
    for(int r = 0; r < this->image.rows; ++r){
        for(int c = 0; c < this->image.cols; ++c){
            int label = labelImage.at<int>(r, c);
            Vec3b &pixel = dstImage.at<Vec3b>(r, c);
            //pixel = colors[label];
            if(label == 0) pixel = Vec3b(0, 0, 0);
            else pixel = Vec3b(255, 255, 255);
        }
    }
    cvtColor(dstImage, this->binImage, CV_BGR2GRAY);
}

vector<Points> ImageExtractor::getContours()
{
    vector<Points> contours, result;
    vector<Vec4i> hierarchy;
    Mat tmpImage;
    //Mat element = getStructuringElement(MORPH_ELLIPSE, Size(15, 15));
    //morphologyEx(this->binImage, tmpImage, MORPH_GRADIENT, element);
    findContours(this->binImage, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0,0));
    for(int i=0;i<contours.size();i++)
    {
        Points obj = fillObject(contours[i]);
        if(obj.size() < 100) continue;
        result.push_back(contours[i]);
    }
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

Vec3b ImageExtractor::getColor(Points obj)
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
    cout << b << " " << g << " "<< r << " " << endl;
    /*
    const int M = 15, N = 5;
    if(b>r+20 && b>g+20) return B;
    if(r>b+10 && r>g+10) return R;
    if((g>b+M && g>r+M) || (g>b+N && g>r+N && g > 200)) return G;
    return Y;
    */
    Vec3b result;
    result[0] = (char)b;
    result[1] = (char)g;
    result[2] = (char)r;
    return result;
}

void ImageExtractor::getColors(vector<Points> objs, char* colors)
{
    memset(colors, 0, sizeof(char)*objs.size());
    vector<Vec3b> bgr;
    for(int i=0;i<objs.size();i++)
    {
        bgr.push_back(this->getColor(objs[i]));
    }
    int nr = 0, ng = 0, nb = 0, ny = 0;
    int bval = 10, rval = 10, gval = 5, yval = 5;
    while((nr < 2 || nb < 2)  && bval >= 0 && rval >=0)
    {
        for(int i=0;i<bgr.size();i++)
        {
            if(colors[i] > 0) continue;
            long b = (long)(bgr[i][0]);
            long g = (long)(bgr[i][1]);
            long r = (long)(bgr[i][2]);
            if(b>g+bval && b > r+bval) 
            {
                colors[i] = B;
                nb++;
            }
            else if(r>b+rval && r>g+rval) 
            {
                colors[i] = R;
                nr++;
            }
        }
        bval--;
        rval--;
    }
    while(ng != 2 && gval >=0)
    {
        ng = 0;
        for(int i=0;i<bgr.size();i++)
        {
            if(colors[i] > 0) continue;
            long b = (long)(bgr[i][0]);
            long g = (long)(bgr[i][1]);
            long r = (long)(bgr[i][2]);
            if(b>r+gval && g>r+gval) ng++;
        }
        gval--;
    }
    gval++;
    for(int i=0;i<bgr.size();i++)
    {
        if(colors[i] > 0) continue;
        long b = (long)(bgr[i][0]);
        long g = (long)(bgr[i][1]);
        long r = (long)(bgr[i][2]);
        if(b>r+gval && g>r+gval) colors[i] = G;
    }

    for(int i=0;i<bgr.size();i++)
    {
        if(colors[i] == 0) colors[i] = Y;
    }
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

bool ImageExtractor::isStand(Points contour, Point centroid)
{
    vector<float> dists;
    for(int i=0;i<contour.size();i++)
    {
        int dx = contour[i].x - centroid.x;
        int dy = contour[i].y - centroid.y;
        dists.push_back(dx * dx + dy * dy);
    }
    float mean = 0.0, var = 0.0;
    for(int i=0;i<dists.size();i++) 
    {
        mean += dists[i];
    }
    mean /= dists.size();
    for(int i=0;i<dists.size();i++)
    {
        int d = dists[i] - mean;
        var += d * d;
    }
    sort(dists.begin(), dists.end());
    float far = 0.0, near = 0.0;
    for(int i=0;i<5;i++)
    { 
        near += dists[i];
        far += dists[dists.size() - 1 - i];
    }
    float ratio = far / near;
    //cout << var << " " << ratio<< endl;
    return ratio < 3 && var < 1e7;
}

int ImageExtractor::connectedComponents(Mat &L, const Mat &I, int connectivity){
    CV_Assert(L.rows == I.rows);
    CV_Assert(L.cols == I.cols);
    CV_Assert(L.channels() == 1 && I.channels() == 1);
    CV_Assert(connectivity == 8 || connectivity == 4);

    int lDepth = L.depth();
    int iDepth = I.depth();
    //using connectedcomponents::LabelingImpl;
    //warn if L's depth is not sufficient?

    if(lDepth == CV_8U){
        if(iDepth == CV_8U || iDepth == CV_8S){
            if(connectivity == 4){
                return LabelingImpl<uint8_t, uint8_t, 4>()(L, I);
            }else{
                return LabelingImpl<uint8_t, uint8_t, 8>()(L, I);
            }
        }else if(iDepth == CV_16U || iDepth == CV_16S){
            if(connectivity == 4){
                return LabelingImpl<uint8_t, uint16_t, 4>()(L, I);
            }else{
                return LabelingImpl<uint8_t, uint16_t, 8>()(L, I);
            }
        }else if(iDepth == CV_32S){
            if(connectivity == 4){
                return LabelingImpl<uint8_t, int32_t, 4>()(L, I);
            }else{
                return LabelingImpl<uint8_t, int32_t, 8>()(L, I);
            }
        }else if(iDepth == CV_32F){
            if(connectivity == 4){
                return LabelingImpl<uint8_t, float, 4>()(L, I);
            }else{
                return LabelingImpl<uint8_t, float, 8>()(L, I);
            }
        }else if(iDepth == CV_64F){
            if(connectivity == 4){
                return LabelingImpl<uint8_t, double, 4>()(L, I);
            }else{
                return LabelingImpl<uint8_t, double, 8>()(L, I);
            }
        }
    }else if(lDepth == CV_16U){
        if(iDepth == CV_8U || iDepth == CV_8S){
            if(connectivity == 4){
                return LabelingImpl<uint16_t, uint8_t, 4>()(L, I);
            }else{
                return LabelingImpl<uint16_t, uint8_t, 8>()(L, I);
            }
        }else if(iDepth == CV_16U || iDepth == CV_16S){
            if(connectivity == 4){
                return LabelingImpl<uint16_t, uint16_t, 4>()(L, I);
            }else{
                return LabelingImpl<uint16_t, uint16_t, 8>()(L, I);
            }
        }else if(iDepth == CV_32S){
            if(connectivity == 4){
                return LabelingImpl<uint16_t, int32_t, 4>()(L, I);
            }else{
                return LabelingImpl<uint16_t, int32_t, 8>()(L, I);
            }
        }else if(iDepth == CV_32F){
            if(connectivity == 4){
                return LabelingImpl<uint16_t, float, 4>()(L, I);
            }else{
                return LabelingImpl<uint16_t, float, 8>()(L, I);
            }
        }else if(iDepth == CV_64F){
            if(connectivity == 4){
                return LabelingImpl<uint16_t, double, 4>()(L, I);
            }else{
                return LabelingImpl<uint16_t, double, 8>()(L, I);
            }
        }
    }else if(lDepth == CV_32S){
        if(iDepth == CV_8U || iDepth == CV_8S){
            if(connectivity == 4){
                return LabelingImpl<int32_t, uint8_t, 4>()(L, I);
            }else{
                return LabelingImpl<int32_t, uint8_t, 8>()(L, I);
            }
        }else if(iDepth == CV_16U || iDepth == CV_16S){
            if(connectivity == 4){
                return LabelingImpl<int32_t, uint16_t, 4>()(L, I);
            }else{
                return LabelingImpl<int32_t, uint16_t, 8>()(L, I);
            }
        }else if(iDepth == CV_32S){
            if(connectivity == 4){
                return LabelingImpl<int32_t, int32_t, 4>()(L, I);
            }else{
                return LabelingImpl<int32_t, int32_t, 8>()(L, I);
            }
        }else if(iDepth == CV_32F){
            if(connectivity == 4){
                return LabelingImpl<int32_t, float, 4>()(L, I);
            }else{
                return LabelingImpl<int32_t, float, 8>()(L, I);
            }
        }else if(iDepth == CV_64F){
            if(connectivity == 4){
                return LabelingImpl<int32_t, double, 4>()(L, I);
            }else{
                return LabelingImpl<int32_t, double, 8>()(L, I);
            }
        }
    }

    CV_Error(CV_StsUnsupportedFormat, "unsupported label/image type");
    return -1;
}

