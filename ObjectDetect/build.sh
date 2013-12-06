#!/bin/bash
# Adjust CMAKE_MODULE_PATH to find FindOpenCV.cmake or
# to the path where OpenCVConfig.cmake or opencv-config.cmake locates
export OpenCV_DIR=/home/thomas/build/misc/openCV/2.4.7/release
cmake .
make
