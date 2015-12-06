#include "mpi.h"
#include <opencv2/opencv.hpp>

/**
 * Run an OpenCV operation that doesn't require border padding on an image.
 */
void noBorderOp(cv::Mat src, cv::Mat dst);
/**
 * Run an OpenCV operation that does require border padding on an image.
 */
void borderOp(cv::Mat src, cv::Mat dst);

int main(int argc, char **argv)
{
    int rank;
    MPI_Status status;

    MPI_Init( &argc, &argv );
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank == 0)
    {
    }
    else
    {
    }

    MPI_Finalize();
    return 0;
}

void noBorderOp(cv::Mat src, cv::Mat dst)
{
    // Convert the input image to grayscale
    cv::cvtColor(src, dst, CV_BGR2GRAY);
}

void borderOp(cv::Mat src, cv::Mat dst)
{
    // Declare arguments
    cv::Mat kernel;
    cv::Point anchor;
    double delta;
    int ddepth;
    int kernel_size;

    // Initialize arguments for the filter
    anchor = cv::Point( -1, -1 );
    delta = 0;
    ddepth = -1;
    kernel_size = 3;
    kernel = cv::Mat::ones( kernel_size, kernel_size, CV_32F )/ (float)(kernel_size*kernel_size);

    // Run a 2d filter on the src image
    cv::filter2D(src, dst, ddepth , kernel, anchor, delta, cv::BORDER_DEFAULT);
}

