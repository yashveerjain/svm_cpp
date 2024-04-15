
#include <iostream>
#include<vector>
#include<tuple>
#include<cstdlib>

#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Core>

#include "svm.hpp"
#include "data_utils.hpp"

using std::cout;

// std::tuple<std::vector<std::vector<int>, std::vector<std::> custom_ai::LinearSVM::loss(std::vector<std::vector<int>> pred, std::vector<int> target)
std::tuple<custom_ai::MatrixXd_f, int> custom_ai::LinearSVM::loss(custom_ai::MatrixXd_f x, custom_ai::MatrixXd_f pred, custom_ai::VectorXd_i target){
    // Eigen::Matrix<double, _w.rows(),_w.cols()> grad_w;
    custom_ai::MatrixXd_f grad_w = custom_ai::MatrixXd_f::Zero(_w.rows(),_w.cols());
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

void custom_ai::LinearSVM::train(int epoch, float lr, int batch_size){
    int N = _x.rows();
    custom_ai::MatrixXd_f grad_w;
    double total_loss, avg_loss, loss;
    for (int i =1 ;i <= epoch;i++){
        std::vector<int> ind(batch_size,0);
        std::srand(time(0));
        std::generate(ind.begin(), ind.end(), [&N]() {return rand()%N;});

        custom_ai::MatrixXd_f x_batch(batch_size,_x.cols());
        custom_ai::VectorXd_i y_batch(batch_size,1);

        x_batch = _x(ind, Eigen::all);
        y_batch = _y(ind,1);
        // std::cout<<"Rows : " << x_batch.rows();

        custom_ai::MatrixXd_f pred = _x*_w;

        // std::cout<<"pred Rows : " << pred.rows()<<"pred Cols : " << pred.cols();
        std::tie(grad_w,loss) = this->loss(x_batch, pred, y_batch);
        // std::cout<<"loss : " << loss<<std::endl;
        custom_ai::MatrixXd_f w_temp = _w - lr*grad_w;
        // cout<<"weights : "<<_w.mean()<<std::endl;
        // cout<<"Difference between weights : "<<(w_temp-_w).mean();
        // cout<<"Differential weights : "<<(grad_w).mean();
        total_loss += loss;
        _w = w_temp;
        
        if (i%10==0){
            std::cout<<"Loss : "<<total_loss/i<<std::endl;
        }
    }
}