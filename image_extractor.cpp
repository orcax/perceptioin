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
    //this->output = Mat(image.size(), CV_32S); 
    this->bgr = Scalar(255, 255, 255);
}

vector<ImageObject> ImageExtractor::extract()
{
    vector<ImageObject> imageObjects;
    this->binarize();
    vector<Points> contours = this->getContours();
    for(int i=0;i<contours.size();i++)
    {
        ImageObject io;
        io.centroid = this->getCentroid(contours[i]);
        io.orient = this->getOrientation(contours[i]);
        io.color = this->getColor(contours[i]);
        io.stand = this->isStand(contours[i], io.centroid);
        if(io.stand) io.orient = 0.0;
        imageObjects.push_back(io);

        Points obj = this->fillObject(contours[i]);
        this->setOutputColor(io.color);
        this->plotPoints(obj);
        if(!io.stand)
        {
            this->setOutputColor('x');
            this->plotLine(io.orient, io.centroid);
        }
        //this->showImage(this->output, "ccc");
    }
    return imageObjects;
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

void ImageExtractor::binarize()
{
    const int thresh = 190;
    const int erosion_type = MORPH_RECT;

    Mat grayImage, binImage, erodeImage;

    //TODO change erosion
    cvtColor(this->image, grayImage, CV_BGR2GRAY);
    //GaussianBlur(this->grayImage, this->grayImage, Size(9, 9), 2, 2);
    //Canny(this->grayImage, this->binImage, 0, 50, 5); 
    threshold(grayImage, binImage, thresh, 255, 0);
    Mat element = getStructuringElement(erosion_type, Size(15, 15));
    morphologyEx(binImage, erodeImage, MORPH_OPEN, element);

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
    cout << b << " " << g << " "<< r << " " << endl;
    const int M = 20;
    if((b>r+M && b>g+M) || (b>r && b>g && b > 230)) return B;
    if((g>b+M && g>r+M) || (g>b && g>r && g > 222)) return G;
    if((r>b+M && r>g+M) || (r>b && r>g && r > 230)) return R;
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

bool ImageExtractor::isStand(Points contour, Point centroid)
{
    const int radius = 50;
    int near = radius * radius, far = radius * radius;
    for(int i=0;i<contour.size();i++)
    {
        int dx = contour[i].x - centroid.x;
        int dy = contour[i].y - centroid.y;
        int dist = dx * dx + dy * dy;
        near = near > dist ? dist : near;
        far = far < dist ? dist : far;
    }
    //cout << far << " " << near << " " << far - near << endl;
    return far - near < 10000;
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

