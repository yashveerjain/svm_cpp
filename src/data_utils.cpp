

#include<iostream>
#include<string>
#include<filesystem>
// #include<iostream>
#include<fstream>

#include <opencv2/opencv.hpp>

#include "data_utils.hpp"

namespace fs = std::filesystem;

/**
 * @brief Read data from a CIFAR-10 binary file and return as a tuple of matrices (data and target)
 * 
 * @param filepath The file path to the CIFAR-10 binary file
 * @return std::tuple<custom_ai::MatrixXd_f, custom_ai::VectorXd_i> Tuple containing data matrix and target vector
 */
std::tuple<custom_ai::MatrixXd_f, custom_ai::VectorXd_i> custom_ai::Cifar10::read(std::string filepath){
    // Check if the file exists
    fs::path fpath(filepath);
    if (fs::exists(fpath)){
        // Proceed only if the file has a .bin extension
        if (fpath.extension() == ".bin") {
            // Open the file in binary mode for reading
            std::ifstream file(fpath.string(), std::ios_base::binary | std::ios_base::in);

            /*
                Data in the file contains labels and image pixel values in a single row.
                Each row has the following structure:
                <label><red_pixels-1024><green_pixels-1024><blue_pixels-1024>
                Total lines per file is 10000.

                The size of each channel is 1024 pixels.
            */

            int per_channel_size = _img_height * _img_width; // Size of each channel (per channel)
            custom_ai::MatrixXd_f data(_per_bin_sample_size, per_channel_size * _channels); // Matrix to store image data
            custom_ai::VectorXd_i target_array(_per_bin_sample_size, 1); // Vector to store labels

            for (int i = 0; i < 10000; i++){ // Iterate over each row in the file (total 10000 rows)
                std::uint8_t label; // Variable to store label
                std::uint8_t red[per_channel_size]; // Array to store red channel pixel values
                std::uint8_t green[per_channel_size]; // Array to store green channel pixel values
                std::uint8_t blue[per_channel_size]; // Array to store blue channel pixel values
                
                // Read label (1 byte)
                file.read(reinterpret_cast<char*>(&label), sizeof(label));
                // Read red channel pixels (1024 bytes)
                file.read(reinterpret_cast<char*>(&red), per_channel_size);
                // Read green channel pixels (1024 bytes)
                file.read(reinterpret_cast<char*>(&green), per_channel_size);
                // Read blue channel pixels (1024 bytes)
                file.read(reinterpret_cast<char*>(&blue), per_channel_size);

                // Store label in target array
                target_array[i] = label;

                // Store pixel values in the data matrix
                for (int j = 0; j < per_channel_size * 3; j++){
                    if (j < per_channel_size) 
                        data(i, j) = (float)red[j]; // Store red channel pixel values
                    else if (j < per_channel_size * 2) 
                        data(i, j) = (float)green[j - per_channel_size]; // Store green channel pixel values
                    else 
                        data(i, j) = (float)blue[j - per_channel_size * 2]; // Store blue channel pixel values
                }
            }
            // Return data and target as a tuple
            return std::make_tuple(data, target_array);
        }
        // Throw an error if the file does not have a .bin extension
        throw std::runtime_error("Filepath not found: " + filepath);
    }
}


/**
 * @brief Convert data row into an OpenCV image (cv::Mat)
 * 
 * @param data_row The input data row representing an image
 * @return cv::Mat The OpenCV image (cv::Mat) converted from the input data row
 */
cv::Mat custom_ai::Cifar10::getImageFromData(custom_ai::VectorXd_f data_row){

    // Convert the format from float to int for data access
    custom_ai::VectorXd_i data_row_int = data_row.cast<int>();

    // Create an OpenCV image (cv::Mat) with dimensions specified by _img_height and _img_width, and 3 channels (RGB)
    cv::Mat image(_img_height, _img_width, CV_8UC3);

    // Calculate the size of each channel (per_channel_size)
    int per_channel_size = _img_height * _img_width;

    // Iterate over each pixel in the image
    for (int i = 0; i < per_channel_size; i++){
        // Calculate the row and column indices for accessing the pixel
        int row_idx = i / _img_height;
        int col_idx = i % _img_width;
        
        // Assign pixel values in BGR format (blue -> green -> red)
        // Access data_row_int using appropriate indices to extract pixel values for each channel
        // The data_row is organized as follows: [R1, R2, ..., R32, G1, G2, ..., G32, B1, B2, ..., B32]
        // Each channel has per_channel_size elements, with R channel first, followed by G channel, and then B channel
        image.at<cv::Vec3b>(row_idx, col_idx)[0] = data_row[i + per_channel_size * 2]; // Blue channel
        image.at<cv::Vec3b>(row_idx, col_idx)[1] = data_row[i + per_channel_size * 1]; // Green channel
        image.at<cv::Vec3b>(row_idx, col_idx)[2] = data_row[i]; // Red channel
    }

    return image; // Return the OpenCV image (cv::Mat)
}


void custom_ai::showImage(int label, cv::Mat image){
    
    std::cout << "Resized image shape (method 1): " << image.size() << std::endl;
    cv::Mat resized_image;
    cv::resize(image, resized_image, cv::Size(400, 400));

    // Create a window
    cv::namedWindow("Display Window");

    // Display the image
    cv::imshow("Display Window", resized_image);

    // Wait for a key press (optional)
    cv::waitKey(0);

    // Close the window
    cv::destroyWindow("Display Window");
}

