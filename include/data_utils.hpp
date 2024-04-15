#pragma once

#include<string>

#include <eigen3/Eigen/Dense>
#include <opencv2/opencv.hpp>

namespace custom_ai{
    typedef Eigen::MatrixXd MatrixXd_f;
    typedef Eigen::MatrixXd MatrixXd_f;
    typedef Eigen::MatrixXi MatrixXd_i;
    // typedef Eigen::Matrix<std::uint8_t,Eigen::Dynamic, Eigen::Dynamic>    
    typedef Eigen::Matrix<double, Eigen::Dynamic, 1> VectorXd_f;
    typedef Eigen::Matrix<int, Eigen::Dynamic, 1> VectorXd_i;

    // void show image and label
    void show_image(int label, cv::Mat image);

    /**
     * @brief Will handle the data for cifar 10 dataset
     * 
     * description: will able to read and show the data.
     */
    class Cifar10{
        // std::string _cifar10_data_path;
        int _per_bin_sample_size = 10000;
        int _img_height = 32;
        int _img_width = 32;
        int _channels = 3;

      public:
        Cifar10(){};
        // Cifar10() = delete;
        std::tuple<MatrixXd_f, VectorXd_i> read(std::string filepath);
        cv::Mat get_image_from_data(VectorXd_f row);
    };

    // void random_number_generator(std::vector<int>&vec);
}

