#pragma once

#include<string> // Include necessary libraries

#include <eigen3/Eigen/Dense> // Include Eigen library for linear algebra operations
#include <opencv2/opencv.hpp> // Include OpenCV library for image processing

namespace custom_ai {
    // Define custom data types using Eigen library
    typedef Eigen::MatrixXf MatrixXd_f; // Matrix of floats
    typedef Eigen::MatrixXi MatrixXd_i; // Matrix of integers
    typedef Eigen::Matrix<float, Eigen::Dynamic, 1> VectorXd_f; // Vector of floats
    typedef Eigen::Matrix<int, Eigen::Dynamic, 1> VectorXd_i; // Vector of integers

    // Function to show image and its label
    void showImage(int label, cv::Mat image);

    /**
     * @brief Handles data for the CIFAR-10 dataset
     */
    class Cifar10 {
        // Constants defining properties of CIFAR-10 dataset
        const int _per_bin_sample_size = 10000; // Number of samples per binary file
        const int _img_height = 32; // Height of each image
        const int _img_width = 32; // Width of each image
        const int _channels = 3; // Number of channels (RGB) in each image
        const int _nClasses = 10; // Total number of classes in CIFAR-10 dataset

    public:
        /**
         * @brief Constructor for CIFAR-10 class
         */
        Cifar10(){}; // Default constructor

        // Deleted constructor (disabled)
        // Cifar10() = delete;

        /**
         * @brief Reads CIFAR-10 data from a binary file
         * 
         * @param filepath Path to the binary file containing CIFAR-10 data
         * @return std::tuple<MatrixXd_f, VectorXd_i> A tuple containing the data matrix and corresponding labels
         */
        std::tuple<MatrixXd_f, VectorXd_i> read(std::string filepath);

        /**
         * @brief Converts a row of data into an image matrix
         * 
         * @param row Vector representing a row of data
         * @return cv::Mat The image matrix
         */
        cv::Mat getImageFromData(VectorXd_f row);

        /**
         * @brief Gets the feature size of each image
         * 
         * @return int The feature size
         */
        int getFeatureSize() { return _img_height * _img_width * _channels; }

        /**
         * @brief Gets the total number of classes in the dataset
         * 
         * @return int The total number of classes
         */
        int getTotalClasses() { return _nClasses; }
    };

    // Function to generate random numbers (not used in the provided code)
    // void random_number_generator(std::vector<int>&vec);
}

