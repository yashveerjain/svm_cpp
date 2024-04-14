
#include <iostream>
#include<vector>
#include<tuple>
#include<cstdlib>

#include <eigen3/Eigen/Dense>
// #include <eigen3/Eigen/Core>

#include "svm.hpp"
#include "data_utils.hpp"

// std::tuple<std::vector<std::vector<int>, std::vector<std::> custom_ai::LinearSVM::loss(std::vector<std::vector<int>> pred, std::vector<int> target)
std::tuple<custom_ai::MatrixXd_f, int> custom_ai::LinearSVM::loss(custom_ai::MatrixXd_f x, custom_ai::MatrixXd_f pred, custom_ai::VectorXd_i target){
    // Eigen::Matrix<double, _w.rows(),_w.cols()> grad_w;
    custom_ai::MatrixXd_f grad_w(_w.rows(),_w.cols());
    double loss=0,temp_loss=0; 
    int N = target.rows(); 
    for (int i =0; i<target.rows();i++){
        double correct_score = pred(i,target[i]);
        for (int j=0; j<pred.cols();j++)
        {   
            if (j==target[i]){
                for(int r =0;r<_w.rows();r++)grad_w(r,j)+=_x(i,j)/N;
                continue;
            }
            temp_loss = 1 - (pred(i,j)-correct_score) ;
            if (temp_loss > 0){
                loss += temp_loss;
                for(int r =0;r<_w.rows();r++)grad_w(r,j)-=x(i,j)/N;
            }
        } 
    }
    loss/=N;
    return std::make_tuple(grad_w, loss);
}

void custom_ai::LinearSVM::train(int epoch, int lr, int batch_size){
    int N = _x.rows();
    custom_ai::MatrixXd_f grad_w;
    double total_loss, avg_loss, loss;
    for (int i =1 ;i <= epoch;i++){
        std::vector<int> ind(batch_size,0);
        std::srand(time(0));
        std::generate(ind.begin(), ind.end(), [&N]() {return rand()%N;});
        custom_ai::MatrixXd_f x_batch;
        custom_ai::VectorXd_i y_batch;
        for(int j=0;j<ind.size();j++){
            y_batch[j] = _y[ind[j]]; 
            x_batch.row(j) = _x.row(ind[j]);
        }

        custom_ai::VectorXd_f pred = _x*_w;

        std::tie(grad_w,loss) = this->loss(x_batch, pred, y_batch);

        _w -= lr*grad_w;
        total_loss += loss;
        if (i%10==0){
            std::cout<<"Loss : "<<total_loss/i<<std::endl;
        }
    }
}