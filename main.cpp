#include <iostream>
#include <thread>
#include <csignal>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>

#include "ThreadPool.h"

#define CAPTURE_WIDTH 4032
#define CAPTURE_HEIGHT 3040

#define FRAMES 1000
#define FRAMES_PER_SAVE 100
#define FRAME_RATE 30

const int SAVES =  (FRAMES + FRAMES_PER_SAVE - 1)/(FRAMES_PER_SAVE);
bool finished = false;

typedef std::pair<cv::Mat, cv::Mat> image_pair;

void handle(int signal_number)
{
    std::cout << "Signal received" << std::endl;
    finished = true;
}

std::string init_params(int sensor_id)
{
    return "nvarguscamerasrc sensor-id=" + std::to_string(sensor_id) + " ! video/x-raw(memory:NVMM), width=(int)"
           + std::to_string(CAPTURE_WIDTH) + ", height=(int)" + std::to_string(CAPTURE_HEIGHT) + ", framerate=(fraction)" + std::to_string(FRAME_RATE) + "/1 ! "
           + "nvvidconv flip-method=0 ! video/x-raw, width=(int)" + std::to_string(CAPTURE_WIDTH) + ", height=(int)" + std::to_string(CAPTURE_HEIGHT)
           + ", format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink";
}


int main()
{

    signal(SIGINT, handle); // signal to stop recording
    cv::VideoCapture cam0(init_params(0), cv::CAP_GSTREAMER), cam1(init_params(1), cv::CAP_GSTREAMER);

    int saves = 0, q;
    ThreadPool pool(std::thread::hardware_concurrency()-1); // maybe leave a thread for something else. not too sure
    cv::namedWindow("Video 0", cv::WINDOW_NORMAL);
    cv::namedWindow("Video 1", cv::WINDOW_NORMAL);

    while (!finished && saves < SAVES)
    {
        std::vector<image_pair> data; // captures this save
        int count = 0;
        for (int i=0; i < FRAMES_PER_SAVE && !finished; i++, count++)
        {
            image_pair pair;
            cam0.grab();
            cam1.grab();

            cam0.retrieve(pair.first);
            cam1.retrieve(pair.second);
            data.push_back(pair);
        }
        // show the first image of every save for convenience.
        cv::imshow("Video 0", data[0].first);
        cv::imshow("Video 1", data[0].second);

        pool.enqueue([count, saves, data] {
            //
            int start = FRAMES_PER_SAVE*saves;
            for (int i = start ; i < start + count ; i++)
            {
                cv::imwrite("Images/Unfiltered/unfiltered_" + std::to_string(i) + ".jpg", data[i-start].first);
                cv::imwrite("Images/Optical/filtered_" + std::to_string(i) + ".jpg", data[i-start].second);
            }
        });
        saves++;
    }

    cv::destroyAllWindows();
    cam0.release();
    cam1.release();
}

