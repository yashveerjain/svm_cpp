#pragma once


#include <eigen3/Eigen/Dense>


namespace custom_ai{
    typedef Eigen::MatrixXd MatrixXd_f;
    typedef Eigen::Matrix<double, Eigen::Dynamic, 1> VectorXd_f;
    typedef Eigen::Matrix<int, Eigen::Dynamic, 1> VectorXd_i;

    // void random_number_generator(std::vector<int>&vec);
}

