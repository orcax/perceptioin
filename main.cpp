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
    vector<ImageObject> imagedata; 
    ie.extract(imagedata);
    //ie.showImage(ie.image, "aaa");
    ie.showImage(ie.output, "ccc");
    //imwrite("./test.png", ie.image);

    return 0;
}
