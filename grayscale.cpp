#include <opencv2/opencv.hpp>

using namespace cv;

Mat src, src_gray;

int main(int argc, char** argv)
{
  // Read in the source image.
  src = imread(argv[1], 1);
 
  // Convert the source image to grayscale.
  cvtColor( src, src_gray, CV_BGR2GRAY );
}
