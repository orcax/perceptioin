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
Mat binarize(Mat image)
{
    Mat image_gray, image_binary;
    cvtColor(image, image_gray, CV_BGR2GRAY);
    threshold(image_gray, image_binary, 200, 255, 0);
    return image_binary;
}

/** Return list of contours made of set of points;
 */
vector<vector<Point> > get_contours(Mat image)
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
    morphologyEx(image, image_dst, MORPH_OPEN, element);
    findContours(image_dst, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0,0));
    return contours;
}

/** Return filled object given contour.
 */
vector<Point> fill_object(Mat image, vector<Point> contour)
{
    Mat tmp = Mat::zeros(image.size(), CV_8UC3);
    vector<vector<Point> > contours;
    contours.push_back(contour);
    const int mask = 1;
    drawContours(tmp, contours, 0, Scalar(mask, mask, mask), CV_FILLED);
    vector<Point> shape;
    for(int x=0;x<tmp.cols;x++)
    {
        for(int y=0;y<tmp.rows;y++)
        {
            Vec3b c = tmp.at<Vec3b>(y, x);
            if(c[0] == mask)
            {
                shape.push_back(Point(x, y));
            }
        }
    }
    return shape;
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

Point centroid(vector<Point> contour)
{
    Moments mu = moments(contour, false);
    Point mc = Point(mu.m10/mu.m00, mu.m01/mu.m00);
    return mc;
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
    Mat output = Mat::zeros(image_binary.size(), CV_8UC3);
    vector<vector<Point> > contours = get_contours(image_binary);
    vector<vector<Point> > objects;
    for(int i=0;i<contours.size();i++)
    {
        vector<Point> contour = contours[i];
        vector<Point> obj = fill_object(image, contour);
        objects.push_back(obj);
        int color = get_color(image, obj);
        Vec3b brg = get_brg(color);
        Point center = centroid(contour);
        for(int j=0;j<obj.size();j++)
        {
            int x = obj[j].x;
            int y = obj[j].y;
            output.at<Vec3b>(y, x)[0] = brg[0];
            output.at<Vec3b>(y, x)[1] = brg[1];
            output.at<Vec3b>(y, x)[2] = brg[2];
        }
        circle(output, center, 4, color, -1, 8, 0);

        cout << obj.size() << endl;
    }

    namedWindow("Display1", CV_WINDOW_NORMAL);
    imshow("Display1", image);
    namedWindow("Display2", CV_WINDOW_NORMAL);
    imshow("Display2", output);
    waitKey(0);
    return 0;
}
