
#pragma once

#include <iostream>
#include<vector>
#include<tuple>
#include<eigen3/Eigen/Core>

#include "data_utils.hpp"

// using namespace Eigen::placeholders;

namespace custom_ai{
    class LinearSVM{
        custom_ai::MatrixXd_f _w;
        custom_ai::MatrixXd_f _x;
        custom_ai::VectorXd_i _y;
        int _feature_size;
        public:
            LinearSVM(int feature_size, int output_size, custom_ai::MatrixXd_f input, custom_ai::VectorXd_i target):
                _w(custom_ai::MatrixXd_f::Random(feature_size,output_size)), 
                _x(input),
                _y(target){}
            LinearSVM():_x(custom_ai::MatrixXd_f::Random(1000,100)), _y(custom_ai::VectorXd_i::Zero(1000,1)),_w(custom_ai::MatrixXd_f::Random(100,10)){}
            void preprocessor();
            void train(int epoch=100, float lr=0.01, int batch_size=32);
            custom_ai::VectorXd_i predict(std::vector<double> test_data);
            std::tuple<custom_ai::MatrixXd_f, int> loss(custom_ai::MatrixXd_f x, custom_ai::MatrixXd_f pred, custom_ai::VectorXd_i target);
    };
}