// Connor Lukan

#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudaimgproc.hpp>

const int CAPTURE_HEIGHT = 3040;
const int CAPTURE_WIDTH = 4032;

// Orientation of cameras: the NoIR cut camera was to the left of the regular one
int main()
{
    cv::Mat NoIR_im = cv::imread("images/unfiltered_40.jpg", cv::IMREAD_GRAYSCALE);
    cv::Mat Optical_im = cv::imread("images/filtered_40.jpg", cv::IMREAD_GRAYSCALE);

    cv::Mat NoIR_edge_map, Optical_edge_map;
    cv::Canny(NoIR_im, NoIR_edge_map, 10, 40); // canny edge detection on NoIR, store in NoIR edge map
    // you can try to mess with the last two parameters but they were solid when I ran it
    cv::Canny(Optical_im, Optical_edge_map, 10, 40);

    int max_dot = 0;
    int max_shift_r = 0;
    int max_shift_c = 0;

    cv::cuda::GpuMat NoIR_gpu(NoIR_edge_map); // special gpu version of opencv's mat object
    cv::cuda::GpuMat Optical_gpu(Optical_edge_map);

    for (int shift_c = 0 ; shift_c < CAPTURE_WIDTH/2; shift_c++)
    {
	std::cout << "Columns shfited: " << shift_c << std::endl;
        for (int shift_r = 0; shift_r < CAPTURE_HEIGHT/2; shift_r++)
        {
            // design choice: camera on the left was just ever so slightly higher than that on the right for the next assumption to be true
            // take the last CAPTURE_HEIGHT - shift_r rows of the image on the left, first CAPTURE_HEIGHT-shift_r rows on img to the right
            // take the last CAPTURE_WIDTH - shift_c cols on the image to the left, first CAPTURE_WIDTH - shift_c cols of img to the right
            cv::cuda::GpuMat cropped_NoIR(NoIR_gpu, cv::Rect(shift_c, shift_r, CAPTURE_WIDTH-shift_c, CAPTURE_HEIGHT-shift_r));
            cv::cuda::GpuMat cropped_Optical(Optical_gpu, cv::Rect(0, 0, CAPTURE_WIDTH - shift_c, CAPTURE_HEIGHT - shift_r));
	   // std::cout << "NoIR rows: " << cropped_NoIR.rows << "\nOptical rows: " << cropped_Optical.rows << "\nNoIR cols: " << cropped_NoIR.cols << "\nOptical cols: " << cropped_Optical.cols << std::endl;

            cv::cuda::GpuMat almost_dot_gpu;
            // math it out on the gpu
            cv::cuda::multiply(cropped_NoIR, cropped_Optical, almost_dot_gpu); // this is no ordinary old matrix multiplication, refer to https://docs.opencv.org/4.x/d8/d34/group__cudaarithm__elem.html#ga497cc0615bf717e1e615143b56f00591

            // get it back to the cpu
            cv::Mat cpu_almost_dot;

            almost_dot_gpu.download(cpu_almost_dot); // copy the gpu version back to the cpu version
            int dot = cv::sum(cpu_almost_dot)[0]; // element wise sum and now we have the dot product

            if (dot > max_dot)
            {
                max_shift_r = shift_r;
                max_shift_c = shift_c;
                max_dot = dot;
            }
        }
    }

    std::cout << "Column shift: " << max_shift_c << std::endl;
    std::cout << "Row shift: " << max_shift_r << std::endl;
    std::cout << "Dot product at the shift: " << max_dot << std::endl;

    return 0;
}
