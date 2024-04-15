

#include<iostream>
#include<string>
#include<filesystem>
// #include<iostream>
#include<fstream>

#include <opencv2/opencv.hpp>

#include "data_utils.hpp"

namespace fs = std::filesystem;

std::tuple<custom_ai::MatrixXd_f, custom_ai::VectorXd_i> custom_ai::Cifar10::read(std::string filepath){
    fs::path fpath(filepath);
    if (fs::exists(fpath)){
        if (fpath.extension()==".bin") // If the file has particular extension then we will proceed further
        {
            //if the file exist then read it.
           std::cout << "Reading filepath : "<< fpath.string();
            std::ifstream file(fpath.string(),std::ios_base::binary | std::ios_base::in);

            /*
                Data contains labels and image in a single row
                and each is unit8 size or a byte, total size per row is 1 + 1024*3
                <label><redpixes-1204><greenpixels-1024><bluepixels-1024>
                ...

                Total lines per file is 10000.
            */

            int per_channel_size = _img_height*_img_width;
            custom_ai::MatrixXd_f data(_per_bin_sample_size,per_channel_size*_channels);
            std::cout <<data.rows()<<std::endl;
            custom_ai::VectorXd_i target_array(_per_bin_sample_size,1);
            for (int i = 0; i<10000;i++)
            {   
                std::uint8_t label;
                std::uint8_t red[per_channel_size];
                std::uint8_t green[per_channel_size];
                std::uint8_t blue[per_channel_size];
                
                // First byte label
                file.read(reinterpret_cast<char*>(&label), sizeof(label));
                // following 1024 bytes red channel
                file.read(reinterpret_cast<char*>(&red), per_channel_size);
                //green channel
                file.read(reinterpret_cast<char*>(&green), per_channel_size);
                //blue channel
                file.read(reinterpret_cast<char*>(&blue), per_channel_size);

                target_array[i] = label;
                for (int j=0;j<per_channel_size*3;j++){
                    if (j<per_channel_size) data(i,j) = (float)red[j];
                    else if (j<per_channel_size*2) data(i,j) = (float)green[j-per_channel_size];
                    else data(i,j) = (float)blue[j-per_channel_size*2];
                }
            }
            return std::make_tuple(data,target_array);
        }
        throw std::runtime_error("Filepath not found : "+filepath);
    }
}

cv::Mat custom_ai::Cifar10::get_image_from_data(custom_ai::VectorXd_f data_row){

    // convert the format from double to int for data access
    custom_ai::VectorXd_i data_row_int = data_row.cast<int>();
    cv::Mat image(_img_height, _img_width, CV_8UC3);
    int per_channel_size = _img_height*_img_width;
    for (int i=0;i<per_channel_size;i++){
        int row_idx = i/_img_height;
        int col_idx = i%_img_width;
        // BGR format insert in this sequence blue -> green -> red 
        image.at<cv::Vec3b>(row_idx, col_idx)[0] = data_row[i+per_channel_size*2];
        image.at<cv::Vec3b>(row_idx, col_idx)[1] = data_row[i+per_channel_size*1];
        image.at<cv::Vec3b>(row_idx, col_idx)[2] = data_row[i];
    }
    return image;
}


void custom_ai::show_image(int label, cv::Mat image){
    
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

