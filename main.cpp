#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>

using namespace cv;
using namespace std;

#define YELLOW 101
#define GREEN 102
#define BLUE 103
#define RED 104

/** Return binarized image of black and white colors.
 */
Mat binarize(Mat image_src)
{
    Mat image_gray, image_binary;
    cvtColor(image_src, image_gray, CV_BGR2GRAY);
    threshold(image_gray, image_binary, 200, 255, 0);
    return image_binary;
}

/** Return list of objects made of set of points.
 */
vector<vector<Point> > segment_objects(Mat image_src)
{
    Mat image_dst;
    vector<vector<Point> > contours;
    vector<Vec4i> hierarchy;

    //Mat dist;
    //distanceTransform(image_src, dist, CV_DIST_L2, 3);
    //normalize(dist, dist, 0, 1.0, NORM_MINMAX);
    //threshold(dist, dist, 0.65, 1.0, CV_THRESH_BINARY);
    //Mat dist_8u;
    //dist.convertTo(dist_8u, CV_8U);
    //findContours(dist_8u, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);

    //Canny(image_src, image_dst, 100, 200, 3);
    Mat element = getStructuringElement(MORPH_RECT, Size(50, 50));
    morphologyEx(image_src, image_dst, MORPH_OPEN, element);
    findContours(image_dst, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0,0));

    vector<vector<Point> > shapes;
    for(int k=0;k<contours.size();k++)
    {
        Mat tmp = Mat::zeros(image_src.size(), CV_8UC3);
        Scalar color = Scalar(255, 255, 255);
        drawContours(tmp, contours, k, color, CV_FILLED);
        vector<Point> shape;
        for(int x=0;x<tmp.cols;x++)
        {
            for(int y=0;y<tmp.rows;y++)
            {
                Vec3b c = tmp.at<Vec3b>(y, x);
                if(c[0] == 255)
                {
                    shape.push_back(Point(x, y));
                }
            }
        }
        shapes.push_back(shape);
    }
    return shapes;
}

/** Return color of the object on the image.
 */
int get_color(Mat image, vector<Point> obj)
{
    Vec3b brg = image.at<Vec3b>(obj[0].y, obj[0].x);
    for(int i=1;i<obj.size();i++)
    {
        Vec3b tmp = image.at<Vec3b>(obj[i].y, obj[i].x);
        brg[0] = (brg[0] * i + tmp[0]) / (i + 1);
        brg[1] = (brg[1] * i + tmp[1]) / (i + 1);
        brg[2] = (brg[2] * i + tmp[2]) / (i + 1);
    }
    if(brg[0] > 195) return BLUE;
    if(brg[1] > 195) return RED;
    if(brg[2] > 195) return GREEN;
    return YELLOW;
}

Vec3b get_brg(int color)
{
    Vec3b brg(0, 0, 0);
    switch(color)
    {
        case BLUE: 
            brg[0] = 255; break;
        case RED: 
            brg[1] = 255; break;
        case GREEN: 
            brg[2] = 255; break;
        case YELLOW: 
            brg[1] = 255; brg[2] = 255; break;
    }
    return brg;
}

int main(int argc, char** argv)
{
    Mat image = imread(argv[1], CV_LOAD_IMAGE_COLOR);
    if(!image.data) 
    {
        cout << "Could not open image" << endl;
        return -1;
    }

    Mat image_binary = binarize(image);
    vector<vector<Point> > objects = segment_objects(image_binary);
    Mat output = Mat::zeros(image_binary.size(), CV_8UC3);
    for(int i=0;i<objects.size();i++)
    {
        vector<Point> obj = objects[i];
        int color = get_color(image, obj);
        Vec3b brg = get_brg(color);
        /*
        unsigned char r = 255 * (rand()/(1.0 + RAND_MAX));
        unsigned char g = 255 * (rand()/(1.0 + RAND_MAX));
        unsigned char b = 255 * (rand()/(1.0 + RAND_MAX));
        for(int j=0;j<contour.size();j++)
        {
            int x = contour[j].x;
            int y = contour[j].y;
            output.at<Vec3b>(y,x)[0] = b;
            output.at<Vec3b>(y,x)[1] = g;
            output.at<Vec3b>(y,x)[2] = r;
        }
        */
        //drawContours(output, contours, i, color, CV_FILLED);
        cout << obj.size() << endl;
        for(int j=0;j<obj.size();j++)
        {
            int x = obj[j].x;
            int y = obj[j].y;
            output.at<Vec3b>(y, x)[0] = brg[0];
            output.at<Vec3b>(y, x)[1] = brg[1];
            output.at<Vec3b>(y, x)[2] = brg[2];
        }
    }

    namedWindow("Display1", CV_WINDOW_NORMAL);
    imshow("Display1", image);
    namedWindow("Display2", CV_WINDOW_NORMAL);
    imshow("Display2", output);
    waitKey(0);
    return 0;
}
