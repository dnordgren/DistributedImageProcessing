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
/**
 * Returns the process the next subimage should be send to.
 */
int getRecvDest();
/**
 * Paritions img into num_chunks subimages.
 */
void partition(cv::Mat img, int num_chunks, cv::Mat *chunks);

int current_process = 1;
int num_processes = 0;

int main(int argc, char **argv)
{
    if (argc != 4)
    {
        std::cout << "error: incorrect arguments\n";
        std::cout << "usage: <img_path> <num_chunks> <operation_type>\n";
        std::cout << "https://youtu.be/Eo-KmOd3i7s\n";
        return 1;
    }

    int num_chunks;
    int size[2];
    cv::Mat img, dst;

    int rank;
    MPI_Status status;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_processes);

    num_chunks = atoi(argv[2]);

    if (rank == 0)
    {
        // read in the image
        char *filepath = argv[1];
        cv::Mat src = cv::imread(filepath, 1);

        // partition the source image into chunks
        cv::Mat *chunks;
        partition(src, num_chunks, chunks);
        
        for (int i = 0; i < num_chunks; i++)
        {
            img = chunks[i];
            // during the first iteration, send the size of the image to process 
            if (i == 0)
            {
                cv::Size s = img.size();
                size[0] = s.height;
                size[1] = s.width;
                MPI_Send(size, 2, MPI_INT, 1, 0, MPI_COMM_WORLD);
                // initialize the output image
                dst.create(size[0], size[1], CV_8UC3);
            }
            // send the image data to other processes
            MPI_Send(img.data, size[0]*size[1]*3, MPI_CHAR, getRecvDest(), 1, MPI_COMM_WORLD);
            MPI_Recv(dst.data, size[0]*size[1]*3, MPI_CHAR, MPI_ANY_SOURCE, 1, MPI_COMM_WORLD, &status);
            char buffer[100];
            sprintf(buffer, "c_%i", i);  
            cv::imwrite(buffer, dst);
        }
    }
    else
    {
        for (int i = 0; i < num_chunks; i++)
        {
            // during the first iteration, initialize the image to process
            if (i == 0)
            {
                MPI_Recv(size, 2, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
                img.create(size[0], size[1], CV_8UC3);
            }
            // receive the image data to process
            MPI_Recv(img.data, size[0]*size[1]*3, MPI_CHAR, 0, 1, MPI_COMM_WORLD, &status);

            // determine the correct operation to run based on the input flag
            int op_type = atoi(argv[3]);
            void (*operation)(cv::Mat, cv::Mat);
            operation = (op_type == 0) ? &noBorderOp : &borderOp;

            // run the operation on the image
            operation(img, dst);

            // send the output back to head
            MPI_Send(dst.data, size[0]*size[1]*3, MPI_CHAR, 0, 1, MPI_COMM_WORLD);
        }
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

int getRecvDest()
{
    if (++current_process >= num_processes)
    {
        current_process = 1;
    }
    return current_process;
}

void partition(cv::Mat img, int num_chunks, cv::Mat *chunks)
{
    // TODO actually partition image
    chunks = &img;
}

