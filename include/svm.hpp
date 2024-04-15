
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
            LinearSVM(int feature_size, int output_size, custom_ai::MatrixXd_f& input, custom_ai::VectorXd_i& target);
            LinearSVM():_x(custom_ai::MatrixXd_f::Random(1000,100)), _y(custom_ai::VectorXd_i::Zero(1000,1)),_w(custom_ai::MatrixXd_f::Random(100,10)){}
            custom_ai::MatrixXd_f preprocessor(custom_ai::MatrixXd_f& input,const int& rows,const int& cols);
            void train(int epoch=100, float lr=0.01, int batch_size=32, float reg=0);
            custom_ai::VectorXd_i predictClassFromScores(custom_ai::MatrixXd_f& score);
            custom_ai::VectorXd_i inference(custom_ai::MatrixXd_f& test_data);
            float loss(custom_ai::MatrixXd_f& grad_w,const custom_ai::MatrixXd_f& x, const custom_ai::MatrixXd_f& pred, const custom_ai::VectorXd_i& target);
            float computeAccuracy(custom_ai::VectorXd_i pred, custom_ai::VectorXd_i target);
    };
}