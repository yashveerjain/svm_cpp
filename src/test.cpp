
#include<iostream>
#include<eigen3/Eigen/Core>
#include<eigen3/Eigen/Dense>

#include "data_utils.hpp"
#include "svm.hpp"

int main(int argc, char *argv[]){
    custom_ai::MatrixXd_f input = custom_ai::MatrixXd_f::Random(100,50);
    printf("Random Feature created with rows %d and cols %d\n", input.rows(), input.cols());
    printf("Value at row %d and col %d, is : %f",1,2,input(10,2));
}