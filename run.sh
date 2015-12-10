#!/bin/bash

cmake . && make 
./main data/scene_1.png result/scene_1.out.png
./main data/scene_1.png data/scene_1.out.png
./main data/scene_2.png data/scene_2.out.png
./main data/scene_3.png data/scene_3.out.png
./main data/scene_4.png data/scene_4.out.png
./main data/scene_5.png data/scene_5.out.png
./main data/scene_6.png data/scene_6.out.png
./main data/scene_7.png data/scene_7.out.png
./main data/scene_8.png data/scene_8.out.png
