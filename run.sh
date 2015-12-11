#!/bin/bash

cmake . && make 
#./main mydata/frame0000.jpg result/frame0000.jpg
./main data/scene_1.png result/scene_1.out.png
./main data/scene_2.png result/scene_2.out.png
./main data/scene_3.png result/scene_3.out.png
./main data/scene_4.png result/scene_4.out.png
./main data/scene_5.png result/scene_5.out.png
./main data/scene_6.png result/scene_6.out.png
./main data/scene_7.png result/scene_7.out.png
./main data/scene_8.png result/scene_8.out.png
