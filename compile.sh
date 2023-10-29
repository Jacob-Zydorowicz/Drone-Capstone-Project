#!/bin/sh

/usr/local/cuda-10.2/bin/nvcc main.cpp kernels.cu yoloPlugins.cpp -I/usr/include/opencv4 -I/usr/include/aarch64-linux-gnu -I/usr/local/cuda-10.2/include -lcudart -lnvinfer -lnvinfer_plugin -lopencv_core -lopencv_videoio -lopencv_imgcodecs -lopencv_imgproc -lopencv_highgui -o run
