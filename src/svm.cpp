
#include <iostream>
#include<vector>
#include<tuple>
#include<cstdlib>

#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Core>

#include "svm.hpp"
#include "data_utils.hpp"

using std::cout;

/**
 * @brief Construct a new custom ai::LinearSVM::LinearSVM object
 * 
 * @param feature_size 
 * @param output_size 
 * @param input 
 * @param target 
 */
custom_ai::LinearSVM::LinearSVM(int feature_size, int output_size, custom_ai::MatrixXd_f& input, custom_ai::VectorXd_i& target){
    // to generate random matrix everytime code run : ref: https://stackoverflow.com/questions/21292881/matrixxfrandom-always-returning-same-matrices
    std::srand(time(0));
    // adding bias inside the weight and so to that add extra feature in the input as well with value one.
    _w = custom_ai::MatrixXd_f::Random(feature_size+1,output_size);

    int rows = input.rows(), cols = input.cols();
    // first col is of value 1, and other will be copied from the original input dataset.
    _x = custom_ai::MatrixXd_f::Ones(input.rows(),input.cols()+1);

    _x.topRightCorner(rows,cols) = preprocessor(input, rows, cols); 
    // _x.block<rows,cols>(0,1) = preprocessor(input, rows, cols);
    _y = target;

    int test_row = 2;
    printf("Input Feature from preprocess input with rows %ld and cols %ld\n", _x.rows(), _x.cols());
    printf("Weight initialized with rows %ld and cols %ld\n", _w.rows(), _w.cols());
    printf("Input target from preprocess label with rows %ld and cols %ld\n", _y.rows(),_y.cols());
    // std::cout << input(Eigen::in,2);
    printf("Value of feature at row %d and col %d, is : %f\n",test_row,2,_x(test_row,2));
    printf("Value of target at row %d and col %d, is : %d\n",test_row,1,_y(test_row));

}

/**
 * @brief 
 * 
 * @param input 
 * @param rows 
 * @param cols 
 * @return custom_ai::MatrixXd_f 
 */
custom_ai::MatrixXd_f custom_ai::LinearSVM::preprocessor(custom_ai::MatrixXd_f& input,const int& rows,const int& cols){
    custom_ai::MatrixXd_f x_preprocess(rows,cols);
    // compute the mean across the dataset and subtract it from the dataset.
    x_preprocess = input -  custom_ai::MatrixXd_f::Ones(x_preprocess.rows(),x_preprocess.cols())*input.mean();
    
    return x_preprocess/255; // to make it in range between (0-1)
}

/**
 * @brief 
 * 
 * @param pred 
 * @param target 
 * @return float 
 */
float custom_ai::LinearSVM::computeAccuracy(custom_ai::VectorXd_i pred, custom_ai::VectorXd_i target){
    /*
        Computing accuracy
    */
   
    float acc = 0;
    // acc = (target==pred).count()
    for (int i=0; i<pred.rows();i++){
        acc += pred[i]==target[i];
    }
    // std::cout<<acc<<std::endl;
    acc/=pred.rows();

    return acc;
}

/**
 * @brief 
 * 
 * @param scores 
 * @return custom_ai::VectorXd_i 
 */
custom_ai::VectorXd_i custom_ai::LinearSVM::predictClassFromScores(custom_ai::MatrixXd_f& scores){
    float acc = 0;
    int idx;
    custom_ai::VectorXd_i class_pred(scores.rows());
    for (int i=0;i<scores.rows();i++){
        scores.row(i).maxCoeff(&idx);
        class_pred[i] = (int)idx;
    }
    return class_pred;
}

/**
 * @brief compute the svm loss loss = sum(max(0, 1 - (score_other_class-score_correct_class))) sum across all  the classes
 * 
 * @param grad_w : gradient of weights
 * @param x : feature input 
 * @param pred: scores predicted by the linear model
 * @param target : correct labels for the feature input
 * @return float : computed loss
 */
float custom_ai::LinearSVM::loss(custom_ai::MatrixXd_f& grad_w, const custom_ai::MatrixXd_f& x, const custom_ai::MatrixXd_f& pred, const custom_ai::VectorXd_i& target){
    // Eigen::Matrix<float, _w.rows(),_w.cols()> grad_w;
    float loss=0,temp_loss=0; 
    int N = target.rows(); 
    
    //loop over the all the samples/predictions
    for (int i =0; i<target.rows();i++){
        // get the score from the indice of the correct class
        float correct_score = pred(i,target[i]);
        // std::cout<<"correct score : "<<correct_score<<std::endl;

        //loop over the all the classes prediction for a given sample
        for (int j=0; j<pred.cols();j++)
        {   
            // if the class is the correct class, then compute the gradient differently
            if (j==target[i]){
                continue;
            }
            // compute the loss (Hinge loss)
            temp_loss = 1 + (pred(i,j)-correct_score) ;
            // std::cout<<"temp loss : "<<temp_loss<<std::endl;
            if (temp_loss > 0){
                loss += temp_loss;
                for(int r =0;r<_w.rows();r++)grad_w(r,j)+=x(i,j);
                for(int r =0;r<_w.rows();r++)grad_w(r,target[i])-=x(i,target[i]);
            }
        } 
    }

    loss/=N;
    grad_w /= N;
    return loss;
}

/**
 * @brief Train the Linear SVM model
 * 
 * @param epoch range > 1, iterate over the dataset to fit model on dataset  
 * @param lr :learning rate, range between (0,1) fast need to fit
 * @param batch_size range > 1 larger batch size good for training, constraint by hardware resources
 * @param reg :regularization, range between (0,1) regularize the weights, its a l2 regularization
 */
void custom_ai::LinearSVM::train(int epoch, float lr, int batch_size, float reg){
    
    int N = _x.rows();
    custom_ai::MatrixXd_f grad_w = custom_ai::MatrixXd_f::Zero(_w.rows(),_w.cols());   
    float total_loss = 0, avg_loss = 0, loss = 0;
    float total_acc = 0, avg_acc = 0;
    custom_ai::MatrixXd_f x_batch(batch_size,_x.cols());
    custom_ai::VectorXd_i y_batch(batch_size,1);


    if (batch_size>0) batch_size = std::min(batch_size, N);
    else batch_size = N;
    custom_ai::VectorXd_i predClasses(batch_size);
    std::cout<<"Start Training -- \n";
    for (int i =1 ;i <= epoch;i++){

        // Generate random indices with size of batch to take random samples from the dataset
        std::vector<int> ind(batch_size,0);
        //seed always unqiue to get random number everytime
        std::srand(time(0));
        // random number generator its an iterator
        std::generate(ind.begin(), ind.end(), [&N]() {return rand()%N;});

        x_batch = _x(ind, Eigen::all);
        y_batch = _y(ind,1);
        // std::cout<<"Rows : " << x_batch.rows();

        custom_ai::MatrixXd_f scores = x_batch*_w;

        // std::cout<<"pred Rows : " << scores.rows()<<"pred Cols : " << scores.cols();
        loss = this->loss(grad_w, x_batch, scores, y_batch);
        //add regularizer
        loss += reg*(_w*_w.transpose()).sum();
        std::cout<<"loss : " << loss<<std::endl;

        // updating the weights 
        // custom_ai::MatrixXd_f w_temp(_w.rows(),_w.cols()); 
        _w = _w - lr*grad_w + 2* reg *_w;
        cout<<"weights : "<<_w.mean()<<std::endl;
        // cout<<"weights temp shape: "<<w_temp.rows()<<","<<w_temp.cols()<<std::endl;
        // cout<<"Difference between weights : "<<(w_temp-_w).mean();
        cout<<"Differential weights : "<<(grad_w).mean()<<std::endl;
        // _w = w_temp;
        
        total_loss += loss;
        avg_loss = (avg_loss+loss)/2;

        predClasses = predictClassFromScores(scores);
        // printf("predClasses row %d | y_batch row %d",predClasses.rows(),y_batch.rows());
        total_acc += computeAccuracy(predClasses,y_batch);
        avg_acc = total_acc/i;

        grad_w.setZero();
        
        if (i%10==0){
            printf("Loss : %f | Accuracy : %f \n",avg_loss, avg_acc);
            // std::cout<<"Loss : "<<avg_loss<<<<std::endl;
        }
    }
}