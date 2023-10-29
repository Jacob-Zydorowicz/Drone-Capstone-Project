#include <iostream>
#include <opencv2/opencv.hpp>
#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include "yoloPlugins.h"
#include <fstream>
#include <chrono>

const int NETWORK_IMG_W = 416;
const int NETWORK_IMG_H = 416;
const int INPUT_DATA_SIZE = 416 * 416 * 3;
const int YOLO_17_OUT_SIZE = 255 * 13 * 13; 
const int YOLO_24_OUT_SIZE = 255 * 26 * 26;


class Logger : public nvinfer1::ILogger
{
public:
    void log(Severity severity, const char* msg) noexcept
    {
	if (severity <= Severity::kWARNING)
	{
	    std::cout << "ERROR: " << msg << std::endl;
	}
    }
	
} logger;


struct DetectionInfo
{
    float input_layer[INPUT_DATA_SIZE];
    float yolo_17[YOLO_17_OUT_SIZE];
    float yolo_24[YOLO_24_OUT_SIZE];
};


struct Frame
{
    DetectionInfo info;
    cv::Mat original, processed;
};


void put_boxes(cv::Mat& img, float& x, float& y, float& w, float& h, int& r, int& c, cv::Scalar color)
{
    float box_dims = 416.0 / 13.0;
    float upper_left_x = box_dims * (c + (x - (w/2.0)));
    float upper_left_y = box_dims * (13 - (r + (y - (h/2.0))));
    float lower_right_x = box_dims * (c + (x + (w/2.0)));
    float lower_right_y = box_dims * (13 - (r + (y + (h/2.0))));
    cv::rectangle(img, cv::Point(upper_left_x, upper_left_y), cv::Point(lower_right_x, lower_right_y), color, 2);
}

void post_process(Frame& mark)
{
    mark.processed = mark.original;
    for (int r = 0 ; r < 13 ; r++)
    {
	for (int c = 0; c < 13; c++)
	{
	    
	    int flattened_idx = (13*r) + c;
	    
	    float prob_human_given_obj_0 = mark.info.yolo_17[(4*13*13) + flattened_idx];
	    float prob_human_given_obj_1 = mark.info.yolo_17[(85*13*13) + (4*13*13) + flattened_idx];
	    float prob_human_given_obj_2 = mark.info.yolo_17[(2*85*13*13) + (4*13*13) + flattened_idx];
	    if (prob_human_given_obj_0 > 0.2)
	    {
		float x_0 = mark.info.yolo_17[flattened_idx];
		float y_0 = mark.info.yolo_17[169 + flattened_idx];
		float w_0 = mark.info.yolo_17[(2*169) + flattened_idx];
		float h_0 = mark.info.yolo_17[(3*13*13) + flattened_idx];
		put_boxes(mark.processed, x_0, y_0, w_0, h_0, r, c, cv::Scalar(0,0,255));
	    }
	    
	    if (prob_human_given_obj_1 > 0.2)
	    {
		float x_1 = mark.info.yolo_17[(85*13*13) + flattened_idx];
		float y_1 = mark.info.yolo_17[(85*13*13) + 169 + flattened_idx];
		float w_1 = mark.info.yolo_17[(85*13*13) + (2*169) + flattened_idx];
		float h_1 = mark.info.yolo_17[(85*13*13) + (3*13*13) + flattened_idx];
		put_boxes(mark.processed, x_1, y_1, w_1, h_1, r, c, cv::Scalar(0,255,0));
	    }
	
	    if (prob_human_given_obj_2 > 0.2)
	    {
		float x_2 = mark.info.yolo_17[(2*85*13*13) + flattened_idx];
		float y_2 = mark.info.yolo_17[(2*85*13*13) + 169 + flattened_idx];
		float w_2 = mark.info.yolo_17[(2*85*13*13) + (2*169) + flattened_idx];
		float h_2 = mark.info.yolo_17[(2*85*13*13) + (3*13*13) + flattened_idx];
		put_boxes(mark.processed, x_2, y_2, w_2, h_2, r, c, cv::Scalar(255,0,0));
	    }

	}
    }
}

void infer(nvinfer1::IExecutionContext* context, DetectionInfo& info)
{
    cudaStream_t cu_stream;
    cudaStreamCreate(&cu_stream);
    void* gpu_buffers[3]; // data, yolo_17, yolo_24, for GPU memory
    
    cudaMalloc(&gpu_buffers[0], INPUT_DATA_SIZE * sizeof(float));
    cudaMalloc(&gpu_buffers[1], YOLO_17_OUT_SIZE * sizeof(float));
    cudaMalloc(&gpu_buffers[2], YOLO_24_OUT_SIZE * sizeof(float));
    
    cudaMemcpyAsync(gpu_buffers[0], info.input_layer, INPUT_DATA_SIZE * sizeof(float), cudaMemcpyHostToDevice, cu_stream); // put host data into device input buffer

    context->enqueue(1, gpu_buffers, cu_stream, nullptr); // forward

    // retrieve output
    cudaMemcpyAsync(info.yolo_17, gpu_buffers[1], YOLO_17_OUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost, cu_stream);
    cudaMemcpyAsync(info.yolo_24, gpu_buffers[2], YOLO_24_OUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost, cu_stream);
    
    // synchronize and kill the stream
    cudaStreamSynchronize(cu_stream);
    cudaStreamDestroy(cu_stream);

    // deallocate buffers on gpu, since we have the out data on the host now
    cudaFree(gpu_buffers[0]);
    cudaFree(gpu_buffers[1]);
    cudaFree(gpu_buffers[2]);
}


DetectionInfo pre_process(cv::Mat& image)
{
    // resize image, flatten it out and put it in the input blob
    DetectionInfo retVal;
    cv::resize(image, image, cv::Size(NETWORK_IMG_W, NETWORK_IMG_H));
    for (int row = 0; row < NETWORK_IMG_H; row++)
    {
	for (int col = 0 ; col < NETWORK_IMG_W; col++)
	{
	    int flattened_idx = (row*NETWORK_IMG_W) + col;
	    cv::Vec3b channels = image.at<cv::Vec3b>(row, col);
	    retVal.input_layer[flattened_idx] = channels[2] / 255.0; // red channel
	    retVal.input_layer[flattened_idx + (NETWORK_IMG_W*NETWORK_IMG_H)] = channels[1] / 255.0; // green channel
	    retVal.input_layer[flattened_idx + (2*NETWORK_IMG_W*NETWORK_IMG_H)] = channels[0] / 255.0; // blue channel
	}
    }
    return retVal;
}


int main()
{
    cv::VideoCapture camera("nvarguscamerasrc ! video/x-raw(memory:NVMM), width=1280, height=720, format=NV12, framerate=(fraction)30/1 ! nvvidconv flip-method=0 ! video/x-raw, width=1280, height=720, format=BGRx ! videoconvert ! video/x-raw, format=BGR ! appsink", cv::CAP_GSTREAMER);

    std::ifstream stream("Detector/Engines/yolov3-tiny.engine", std::ios::binary); // this is a compiled plan file. It's pretty huge, and can't fit in the github repo unfortunatley, but I have it -- Connor
    stream.seekg(0, stream.end); // move to eof
    int sz = stream.tellg();
    stream.seekg(0, stream.beg);
    char* engine_data = new char[sz];
    stream.read(engine_data, sz); // read all of the file
    std::cout << "Made it here" << std::endl;
    nvinfer1::IRuntime* rt = nvinfer1::createInferRuntime(logger);
    nvinfer1::ICudaEngine* engine = rt->deserializeCudaEngine(engine_data, sz);
    delete[] engine_data;

    nvinfer1::IExecutionContext* context = engine->createExecutionContext();
    int total_time = 0;
    int frames_captured = 0;
    for (; frames_captured < 1000; frames_captured++)
    {
	Frame frame;
	auto t0 = std::chrono::system_clock::now();
	camera >> frame.original;
	frame.info = pre_process(frame.original);
	infer(context, frame.info);
	post_process(frame);
	auto t1 = std::chrono::system_clock::now();
	total_time += std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
	cv::imshow("Detections", frame.processed);
	int q = cv::waitKey(110);
	if (q == 'q') break;
    }

    camera.release();
    cv::destroyAllWindows();
    if (frames_captured!=0)
    {
	float seconds = ((float)total_time)/1000.0;
	std::cout << "Average framerate: " << ((float)frames_captured)/seconds << std::endl;
    }
	
    return 0;
    
}
