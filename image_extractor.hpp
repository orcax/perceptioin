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

const char R = 'r';
const char G = 'g';
const char B = 'b';
const char Y = 'y';
const char W = 'w';

typedef struct image_data_
{
    bool stand;
    char color;
    double orient;
    Point centroid;
    Point2d loc;
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

        void binarize();
        vector<Points> getContours();
        Points fillObject(Points contour);
        char getColor(Points obj);
        Vec3b getBGR(int color);
        Point getCentroid(Points contour);
        double getOrientation(Points contour);

        int connectedComponents(Mat &L, const Mat &I, int connectivity);
};


/*****************************************************************************/
/* Following functions are moved from:                                       */
/* http://code.opencv.org/attachments/467/opencv-connectedcomponents.patch   */
/*****************************************************************************/

template<typename LabelT>
inline static
LabelT findRoot(const vector<LabelT> &P, LabelT i){
    LabelT root = i;
    while(P[root] < root){
        root = P[root];
    }
    return root;
}

template<typename LabelT>
inline static
void setRoot(vector<LabelT> &P, LabelT i, LabelT root){
    while(P[i] < i){
        LabelT j = P[i];
        P[i] = root;
        i = j;
    }
    P[i] = root;
}

template<typename LabelT>
inline static
LabelT find(vector<LabelT> &P, LabelT i){
    LabelT root = findRoot(P, i);
    setRoot(P, i, root);
    return root;
}

template<typename LabelT>
inline static
LabelT set_union(vector<LabelT> &P, LabelT i, LabelT j){
    LabelT root = findRoot(P, i);
    if(i != j){
        LabelT rootj = findRoot(P, j);
        if(root > rootj){
            root = rootj;
        }
        setRoot(P, j, root);
    }
    setRoot(P, i, root);
    return root;
}

template<typename LabelT>
inline static
LabelT flattenL(vector<LabelT> &P){
    LabelT k = 1;
    for(size_t i = 1; i < P.size(); ++i){
        if(P[i] < i){
            P[i] = P[P[i]];
        }else{
            P[i] = k; k = k + 1;
        }
    }
    return k;
}

const int G4[2][2] = {{-1, 0}, {0, -1}};
const int G8[4][2] = {{-1, -1}, {-1, 0}, {-1, 1}, {0, -1}};

template<typename LabelT, typename PixelT, int connectivity = 8>
struct LabelingImpl{

    LabelT operator()(Mat &L, const Mat &I){
        const int rows = L.rows;
        const int cols = L.cols;
        size_t nPixels = size_t(rows) * cols;
        vector<LabelT> P; P.push_back(0);
        LabelT l = 1;
        //scanning phase
        for(int r_i = 0; r_i < rows; ++r_i){
            for(int c_i = 0; c_i < cols; ++c_i){
                if(!I.at<PixelT>(r_i, c_i)){
                    L.at<LabelT>(r_i, c_i) = 0;
                    continue;
                }
                if(connectivity == 8){
                    const int a = 0;
                    const int b = 1;
                    const int c = 2;
                    const int d = 3;

                    bool T[4];

                    for(size_t i = 0; i < 4; ++i){
                        int gr = r_i + G8[i][0];
                        int gc = c_i + G8[i][1];
                        T[i] = false;
                        if(gr >= 0 && gr < I.rows && gc >= 0 && gc < I.cols){
                            if(I.at<PixelT>(gr, gc)){
                                T[i] = true;
                            }
                        }
                    }

                    //decision tree
                    if(T[b]){
                        //copy(b)
                        L.at<LabelT>(r_i, c_i) = L.at<LabelT>(r_i + G8[b][0], c_i + G8[b][1]);
                    }else{//not b
                        if(T[c]){
                            if(T[a]){
                                //copy(c, a)
                                L.at<LabelT>(r_i, c_i) = set_union(P, L.at<LabelT>(r_i + G8[c][0], c_i + G8[c][1]), L.at<LabelT>(r_i + G8[a][0], c_i + G8[a][1]));
                            }else{
                                if(T[d]){
                                    //copy(c, d)
                                    L.at<LabelT>(r_i, c_i) = set_union(P, L.at<LabelT>(r_i + G8[c][0], c_i + G8[c][1]), L.at<LabelT>(r_i + G8[d][0], c_i + G8[d][1]));
                                }else{
                                    //copy(c)
                                    L.at<LabelT>(r_i, c_i) = L.at<LabelT>(r_i + G8[c][0], c_i + G8[c][1]);
                                }
                            }
                        }else{//not c
                            if(T[a]){
                                //copy(a)
                                L.at<LabelT>(r_i, c_i) = L.at<LabelT>(r_i + G8[a][0], c_i + G8[a][1]);
                            }else{
                                if(T[d]){
                                    //copy(d)
                                    L.at<LabelT>(r_i, c_i) = L.at<LabelT>(r_i + G8[d][0], c_i + G8[d][1]);
                                }else{
                                    //new label
                                    L.at<LabelT>(r_i, c_i) = l;
                                    P.push_back(l);//P[l] = l;
                                    l = l + 1;
                                }
                            }
                        }
                    }
                }else{
                    //B & D only
                    const int b = 0;
                    const int d = 1;
                    assert(connectivity == 4);
                    bool T[2];
                    for(size_t i = 0; i < 2; ++i){
                        int gr = r_i + G4[i][0];
                        int gc = c_i + G4[i][1];
                        T[i] = false;
                        if(gr >= 0 && gr < I.rows && gc >= 0 && gc < I.cols){
                            if(I.at<PixelT>(gr, gc)){
                                T[i] = true;
                            }
                        }
                    }

                    if(T[b]){
                        if(T[d]){
                            //copy(d, b)
                            L.at<LabelT>(r_i, c_i) = set_union(P, L.at<LabelT>(r_i + G4[d][0], c_i + G4[d][1]), L.at<LabelT>(r_i + G4[b][0], c_i + G4[b][1]));
                        }else{
                            //copy(b)
                            L.at<LabelT>(r_i, c_i) = L.at<LabelT>(r_i + G4[b][0], c_i + G4[b][1]);
                        }
                    }else{
                        if(T[d]){
                            //copy(d)
                            L.at<LabelT>(r_i, c_i) = L.at<LabelT>(r_i + G4[d][0], c_i + G4[d][1]);
                        }else{
                            //new label
                            L.at<LabelT>(r_i, c_i) = l;
                            P.push_back(l);//P[l] = l;
                            l = l + 1;
                        }
                    }

                }
            }
        }

        //analysis
        LabelT nLabels = flattenL(P);

        //assign final labels
        for(size_t r = 0; r < L.rows; ++r){
            for(size_t c = 0; c < L.cols; ++c){
                L.at<LabelT>(r, c) = P[L.at<LabelT>(r, c)];
            }
        }

        return nLabels;
    }

};

#endif
