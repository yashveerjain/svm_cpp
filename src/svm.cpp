
#include <iostream>
#include<vector>
#include<tuple>
#include<cstdlib>
#include<fstream>
#include<iostream>
#include <filesystem>
#include <stdexcept>
#include <time.h>
#include <thread>
#include<mutex>

#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Core>

#include "svm.hpp"
#include "data_utils.hpp"

using std::cout;
namespace fs = std::filesystem;


/**
 * @brief Constructor for the Linear SVM class
 * 
 * @param feature_size The size of input features
 * @param output_size The number of output classes
 * @param input The input feature matrix
 * @param target The target labels
 */
svm::LinearSVM::LinearSVM(int feature_size, int output_size, custom_ai::MatrixXd_f& input, custom_ai::VectorXd_i& target){
    // Seed for random number generation
    std::srand(time(0));
    // Initialize weights with random values using Xavier initialization
    _w = custom_ai::MatrixXd_f::Random(feature_size+1,output_size)/std::sqrt(feature_size);

    int rows = input.rows(), cols = input.cols();
    // Add bias term to the input by adding an extra feature with value one
    _x = custom_ai::MatrixXd_f::Ones(input.rows(),input.cols()+1);

    // Preprocess input data
    _x.topRightCorner(rows,cols) = preprocessor(input, rows, cols); 

    // Store target labels
    _y = target;

    // int test_row = 2; // Just a test row for debugging purposes
}

/**
 * @brief Preprocesses input data
 * 
 * @param input The input feature matrix
 * @param rows Number of rows in the input matrix
 * @param cols Number of columns in the input matrix
 * @return custom_ai::MatrixXd_f Preprocessed input feature matrix
 */
custom_ai::MatrixXd_f svm::LinearSVM::preprocessor(custom_ai::MatrixXd_f& input,const int& rows,const int& cols){
    custom_ai::MatrixXd_f x_preprocess(rows,cols);
    // Subtract mean from each feature to center the data
    x_preprocess = input.rowwise() - input.colwise().mean(); 
    // Normalize the data to be in the range [0, 1]
    return x_preprocess/255; 
}

/**
 * @brief Computes accuracy of predictions
 * 
 * @param pred Predicted labels
 * @param target True labels
 * @return float Accuracy value
 */
float svm::LinearSVM::computeAccuracy(custom_ai::VectorXd_i pred, custom_ai::VectorXd_i target){
    float acc = 0;

    // Calculate accuracy by comparing predicted and true labels
    for (int i=0; i<pred.rows();i++){
        acc += pred[i]==target[i];
    }
    acc/=pred.rows(); // Normalize by the total number of samples
    return acc;
}

/**
 * @brief Predict class labels from the scores predicted by the linear model
 * 
 * @param scores Predicted scores for each class by the linear model
 * @return custom_ai::VectorXd_i Predicted class labels
 */
custom_ai::VectorXd_i svm::LinearSVM::predictClassFromScores(custom_ai::MatrixXd_f& scores){
    int idx; // Variable to store the index of the maximum score
    custom_ai::VectorXd_i class_pred(scores.rows()); // Vector to store predicted class labels

    // scores.rowise().maxCoeff(&idx);
    // Iterate over each sample
    for (int i = 0; i < scores.rows(); i++){
        // Find the index of the maximum score for the current sample
        scores.row(i).maxCoeff(&idx);
        // Assign the index (class label) to the predicted class label
        class_pred[i] = (int)idx;
    }
    
    return class_pred; // Return the vector of predicted class labels
}


/**
 * @brief Compute the SVM loss: loss = sum(max(0, 1 - (score_other_class - score_correct_class))) across all classes
 * 
 * @param grad_w Gradient of weights to be computed and updated during loss calculation
 * @param x Feature input matrix
 * @param pred Predicted scores by the linear model
 * @param target Correct labels for the feature input
 * @return float Computed loss
 */
float svm::LinearSVM::lossNaive(custom_ai::MatrixXd_f& grad_w, const custom_ai::MatrixXd_f& x, const custom_ai::MatrixXd_f& pred, const custom_ai::VectorXd_i& target){
    float loss = 0; // Initialize loss
    int N = target.rows(); // Number of samples

    // Loop over all samples/predictions
    for (int i = 0; i < target.rows(); i++){
        // Get the score from the index of the correct class
        float correct_score = pred(i, target[i]);

        // Loop over all the classes' predictions for a given sample
        for (int j = 0; j < pred.cols(); j++){
            // If the class is the correct class, then skip (continue)
            if (j == target[i]){
                continue;
            }

            // Compute the hinge loss
            float temp_loss = 1 + (pred(i, j) - correct_score);
            if (temp_loss > 0){ // Only consider non-zero losses (hinge loss)
                loss += temp_loss; // Accumulate loss
                grad_w.col(j) += x.row(i); // Update gradient for incorrect class
                grad_w.col(target[i]) -= x.row(i); // Update gradient for correct class
            }
        } 
    }

    // Normalize loss by the number of samples
    loss /= N;
    // Normalize gradient by the number of samples
    grad_w /= N;
    return loss; // Return computed loss
}


/**
 * @brief Compute the SVM loss: loss = sum(max(0, 1 - (score_other_class - score_correct_class))) across all classes
 * Not helpful in svm perform worse then lossNaive.
 * 
 * @param grad_w Gradient of weights to be computed and updated during loss calculation
 * @param x Feature input matrix
 * @param pred Predicted scores by the linear model
 * @param target Correct labels for the feature input
 * @return float Computed loss
 */
float svm::LinearSVM::lossThreadOptimized(custom_ai::MatrixXd_f& grad_w, const custom_ai::MatrixXd_f& x, const custom_ai::MatrixXd_f& pred, const custom_ai::VectorXd_i& target){
    float loss = 0; // Initialize loss
    int N = target.rows(); // Number of samples

    std::vector<std::thread> threadProcesses;
    threadProcesses.reserve(N);

    auto lossCompute = [&grad_w, &x, &pred, &loss, &target](int i){
        // Get the score from the index of the correct class
        float correct_score = pred(i, target[i]);

        // Loop over all the classes' predictions for a given sample
        for (int j = 0; j < pred.cols(); j++){
            // If the class is the correct class, then skip (continue)
            if (j == target[i]){
                continue;
            }

            /**
             * use a mutex lock to protect updates to shared variables.
             * Lockguard : 
             * Its constructor takes as an argument a mutex, which it then locks
             * Its destructor unlocks the mutex
             * Use to safe guard the share variable between threads, so update one at a time.
            */
           
            std::mutex mutex;
            std::lock_guard<std::mutex> lockGuard(mutex);
            // Compute the hinge loss
            float temp_loss = 1 + (pred(i, j) - correct_score);
            if (temp_loss > 0){ // Only consider non-zero losses (hinge loss)
                loss += temp_loss; // Accumulate loss
                grad_w.col(j) += x.row(i); // Update gradient for incorrect class
                grad_w.col(target[i]) -= x.row(i); // Update gradient for correct class
            }
        } 
    };

    // Loop over all samples/predictions
    for (int i = 0; i < target.rows(); i++){
        threadProcesses.push_back(std::thread(lossCompute,i));
    }

    // Loop over all thread processes and join them
    for (auto &threadProcess: threadProcesses){
        threadProcess.join();
    }


    // Normalize loss by the number of samples
    loss /= N;
    // Normalize gradient by the number of samples
    grad_w /= N;
    return loss; // Return computed loss
}

float svm::LinearSVM::lossVectoriesed(custom_ai::MatrixXd_f& grad_w, const custom_ai::MatrixXd_f& x, const custom_ai::MatrixXd_f& pred, const custom_ai::VectorXd_i& target){
    /*
        Incomplete function need to work upon
    */
    
    // Eigen::Matrix<float, _w.rows(),_w.cols()> grad_w;
    float loss=0,temp_loss=0; 
    int N = target.rows(); 

    custom_ai::VectorXd_f correct_scores = pred(Eigen::all, target);
    custom_ai::MatrixXd_f losses = custom_ai::MatrixXd_f::Zero(pred.rows(),pred.cols());
    // custom_ai::MatrixXd_f  = custom_ai::MatrixXd_f::Zero(pred.rows(),pred.cols());
    losses = (pred.colwise() - correct_scores); //custom_ai::MatrixXd_f::Ones(pred.rows(),pred.cols())
    losses.array()+=1;
    losses = losses.cwiseMax(0.0);
    losses(Eigen::all, target).setZero(); //= custom_ai::VectorXd_f::Zero(pred.rows());

    

    loss = losses.sum();


    grad_w = x.transpose()*losses;

    // for (int i=0;i<)
    //     if (temp_loss > 0)
    //     {
    //         loss += temp_loss;
    //         for(int r =0;r<_w.rows();r++)
    //         {   
    //             grad_w(r,j)+=x(i,j)*temp_loss;
    //             // for(int r =0;r<_w.rows();r++)
    //             grad_w(r,target[i])-=x(i,target[i])*temp_loss;
    //         }
    //     }
    loss/=N;
    grad_w /= N;
    return loss;
}


void svm::LinearSVM::genRandomBatch(custom_ai::MatrixXd_f &x_batch, custom_ai::VectorXd_i &y_batch , const int &N, const int &batch_size){
    std::vector<int> ind(batch_size,0);
    std::srand(time(0)); // Seed for random number generation
    std::generate(ind.begin(), ind.end(), [&N]() {return rand() % N;});

        // Select mini-batch input and target samples
    x_batch = _x(ind, Eigen::all);
    y_batch = _y(ind,1);
}

/**
 * @brief Train the Linear SVM model
 * 
 * @param epoch The number of epochs (iterations) to train the model. Should be greater than 1.
 * @param lr The learning rate for updating the model parameters during training. Should be in the range (0, 1).
 * @param batch_size The size of mini-batches used in training. Larger batch sizes are generally good for training efficiency but are constrained by hardware resources.
 * @param reg The regularization parameter. This helps in controlling overfitting by penalizing large weights. Should be in the range (0, 1). It's a form of L2 regularization.
 */
void svm::LinearSVM::train(int epoch, float lr, int batch_size, float reg, bool monitor_time){
    // Number of samples in the dataset
    int N = _x.rows();
    // Initialize gradient of weights to zeros
    custom_ai::MatrixXd_f grad_w = custom_ai::MatrixXd_f::Zero(_w.rows(),_w.cols());   
    // Variables to track total and average loss
    float total_loss = 0, avg_loss = 0, loss = 0;
    // Variables to track total and average accuracy
    float total_acc = 0, avg_acc = 0;
    // Initialize mini-batch input and target matrices
    custom_ai::MatrixXd_f x_batch(batch_size,_x.cols());
    custom_ai::VectorXd_i y_batch(batch_size,1);

    // Ensure batch size is within valid range
    if (batch_size > 0) 
        batch_size = std::min(batch_size, N);
    else 
        batch_size = N;

    // Vector to store predicted classes for each sample in the mini-batch
    custom_ai::VectorXd_i predClasses(batch_size);

    auto total_training_time = 0.0;
    // Start training loop
    std::cout<<"Start Training -- \n";
    for (int i = 1; i <= epoch; i++){

        auto training_s_time = clock();
        auto batch_gen_s_time = clock();
        // Generate random indices to select samples for the mini-batch
        genRandomBatch(x_batch, y_batch, N, batch_size);
        auto batch_gen_e_time = clock();
        
        // Compute scores (predictions) for the mini-batch
        custom_ai::MatrixXd_f scores = x_batch * _w;

        auto loss_s_time = clock();
        // Compute loss and update gradients
        loss = this->lossThreadOptimized(grad_w, x_batch, scores, y_batch);
        loss += reg * (_w * _w.transpose()).sum(); // Apply regularization
        auto loss_e_time = clock();

        _w = _w - lr * grad_w + 2 * reg * _w; // Update weights

        // Update total and average loss
        total_loss += loss;
        avg_loss = (avg_loss + loss) / 2;

        auto compute_acc_s_time = clock();
        // Compute predicted classes and update accuracy metrics
        predClasses = predictClassFromScores(scores);
        total_acc += computeAccuracy(predClasses, y_batch);
        avg_acc = total_acc / i;
        auto compute_acc_e_time = clock();
        // Reset gradient of weights for the next iteration
        grad_w.fill(0.0);
        
        auto training_e_time = clock();

        if (monitor_time){
            double training_time, loss_time, batch_gen_time, compute_acc_time;
            training_time = (double) (training_e_time-training_s_time)/CLOCKS_PER_SEC;
            loss_time = (double) (loss_e_time-loss_s_time)/CLOCKS_PER_SEC;
            batch_gen_time = (double) (batch_gen_e_time-batch_gen_s_time)/CLOCKS_PER_SEC;
            compute_acc_time = (double)(compute_acc_e_time-compute_acc_s_time)/CLOCKS_PER_SEC;
            printf("\n===========\n");
            printf("Training time per epoch : %f\n",training_time);
            printf("Batch Gen time per epoch : %f\n",batch_gen_time);
            printf("Loss time per epoch : %f\n",loss_time);
            printf("Loss : %f | Accuracy : %f \n", avg_loss, avg_acc);
            printf("Compute Accuracy time per epoch : %f\n",compute_acc_time);
            total_training_time += training_time;
        }
        
        // Print loss and accuracy every 10 epochs
        if (i % 10 == 0){
            printf("Loss : %f | Accuracy : %f \n", avg_loss, avg_acc);
        }
    }
    if (monitor_time){
        printf("\nTotal Training time for epochs %d : %f\n",epoch,total_training_time);
    }

}

/**
 * @brief Predicts classes for given test data using the trained Linear SVM model
 * 
 * @param test_data The feature input for which predictions are to be made
 * @return custom_ai::VectorXd_i The predicted classes for the test data
 */
custom_ai::VectorXd_i svm::LinearSVM::predict(custom_ai::MatrixXd_f& test_data){
    // Preprocess the test data by adding a column of ones for bias and scaling
    custom_ai::MatrixXd_f _test_data(test_data.rows(), test_data.cols() + 1); // Initialize matrix for preprocessed test data
    std::cout<<"Rows : "<<test_data.rows()<<" Cols : "<<test_data.cols()<<std::endl;
    
    _test_data.topRightCorner(test_data.rows(), test_data.cols()) = preprocessor(test_data, test_data.rows(), test_data.cols()); // Perform preprocessing

    // Calculate scores for the test data using the trained weights
    custom_ai::MatrixXd_f scores = _test_data * _w;

    // Predict classes from the computed scores
    return predictClassFromScores(scores);
}


void svm::LinearSVM::saveModel(std::string model_path){


    std::ofstream file(model_path, std::ios_base::binary | std::ios_base::out);

    int row = _w.rows(), col = _w.cols();
    // reference : https://eigen.tuxfamily.org/dox/group__TopicStorageOrders.html
    file.write(reinterpret_cast<char*>(&row),sizeof(int));
    file.write(reinterpret_cast<char*>(&col),sizeof(int));
    file.write(reinterpret_cast<char*>(_w.data()), sizeof(float)*_w.size());

    printf("Saving the weight with mean %f in file %s\n",_w.mean(),model_path);
}


void svm::LinearSVM::loadModel(std::string model_path){
    //by default it will assume the weight size initialized during the initialization of the SVM class object.
    // custom_ai::MatrixXd_f mat(_w.rows(),_w.cols());

    if (!fs::exists(model_path)){
        throw std::runtime_error("Model path doesn't exist"+model_path);
    }

    std::ifstream file(model_path, std::ios_base::binary | std::ios_base::in);
    int row,col;
    file.read(reinterpret_cast<char*>(&row),sizeof(int));
    file.read(reinterpret_cast<char*>(&col),sizeof(int));
    custom_ai::MatrixXd_f mat(row,col);
    std::cout<<"Rows : "<<mat.rows()<<" Cols : "<<mat.cols()<<std::endl;
    
    // need to know the size of the weight before hand to get the data from the file.
    file.read(reinterpret_cast<char*>(mat.data()), sizeof(float)*mat.size());

    _w = mat;

    printf("Loading the weight with mean %f from file %s\n",mat.mean(),model_path);
}
