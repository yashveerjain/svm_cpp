
#include <iostream>

#include <Eigen/Dense>
#include <opencv2/opencv.hpp>


namespace custom_ai{
    cv::Mat create_random_RGB_image(int row=480, int col=640);
    cv::Mat create_white_image(int row=480, int col=640);
    cv::Mat creat_black_image(int row=480, int col=640);
    cv::Mat read_image(std::string image_path);

    template <class T>
    class Image{
            typedef cv::Mat_<T> Mat_t; 
            Mat_t _image;
        public:
            cv::Mat read_image(std::string image_path);
            
    }
}