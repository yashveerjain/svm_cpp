
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
 * @brief Constructor for the Linear SVM class
 * 
 * @param feature_size The size of input features
 * @param output_size The number of output classes
 * @param input The input feature matrix
 * @param target The target labels
 */
custom_ai::LinearSVM::LinearSVM(int feature_size, int output_size, custom_ai::MatrixXd_f& input, custom_ai::VectorXd_i& target){
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
custom_ai::MatrixXd_f custom_ai::LinearSVM::preprocessor(custom_ai::MatrixXd_f& input,const int& rows,const int& cols){
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
float custom_ai::LinearSVM::computeAccuracy(custom_ai::VectorXd_i pred, custom_ai::VectorXd_i target){
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
custom_ai::VectorXd_i custom_ai::LinearSVM::predictClassFromScores(custom_ai::MatrixXd_f& scores){
    int idx; // Variable to store the index of the maximum score
    custom_ai::VectorXd_i class_pred(scores.rows()); // Vector to store predicted class labels

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
float custom_ai::LinearSVM::loss(custom_ai::MatrixXd_f& grad_w, const custom_ai::MatrixXd_f& x, const custom_ai::MatrixXd_f& pred, const custom_ai::VectorXd_i& target){
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


float custom_ai::LinearSVM::loss_vectoriesed(custom_ai::MatrixXd_f& grad_w, const custom_ai::MatrixXd_f& x, const custom_ai::MatrixXd_f& pred, const custom_ai::VectorXd_i& target){
    /*
        Incomplete function need to work upon
    */
    
    // Eigen::Matrix<float, _w.rows(),_w.cols()> grad_w;
    float loss=0,temp_loss=0; 
    int N = target.rows(); 

    custom_ai::VectorXd_f correct_scores = pred(Eigen::all, target);
    custom_ai::MatrixXd_f losses = custom_ai::MatrixXd_f::Zero(pred.rows(),pred.cols());
    losses = (pred.colwise() - correct_scores); //custom_ai::MatrixXd_f::Ones(pred.rows(),pred.cols())
    losses.array()+=1;
    losses = losses.cwiseMax(0.0);
    losses(Eigen::all, target).setZero(); //= custom_ai::VectorXd_f::Zero(pred.rows());



    // grad_w.cw
    // for (int i=0;i<)
    // if (temp_loss > 0)
    // {
    //     loss += temp_loss;
    //     for(int r =0;r<_w.rows();r++)
    //     {   
    //         grad_w(r,j)+=x(i,j)*temp_loss;
    //         // for(int r =0;r<_w.rows();r++)
    //         grad_w(r,target[i])-=x(i,target[i])*temp_loss;
    //     }
    // }
    loss/=N;
    grad_w /= N;
    return loss;
}


/**
 * @brief Train the Linear SVM model
 * 
 * @param epoch The number of epochs (iterations) to train the model. Should be greater than 1.
 * @param lr The learning rate for updating the model parameters during training. Should be in the range (0, 1).
 * @param batch_size The size of mini-batches used in training. Larger batch sizes are generally good for training efficiency but are constrained by hardware resources.
 * @param reg The regularization parameter. This helps in controlling overfitting by penalizing large weights. Should be in the range (0, 1). It's a form of L2 regularization.
 */
void custom_ai::LinearSVM::train(int epoch, float lr, int batch_size, float reg){
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

    // Start training loop
    std::cout<<"Start Training -- \n";
    for (int i = 1; i <= epoch; i++){
        // Generate random indices to select samples for the mini-batch
        std::vector<int> ind(batch_size,0);
        std::srand(time(0)); // Seed for random number generation
        std::generate(ind.begin(), ind.end(), [&N]() {return rand() % N;});

        // Select mini-batch input and target samples
        x_batch = _x(ind, Eigen::all);
        y_batch = _y(ind,1);

        // Compute scores (predictions) for the mini-batch
        custom_ai::MatrixXd_f scores = x_batch * _w;

        // Compute loss and update gradients
        loss = this->loss(grad_w, x_batch, scores, y_batch);
        loss += reg * (_w * _w.transpose()).sum(); // Apply regularization
        _w = _w - lr * grad_w + 2 * reg * _w; // Update weights

        // Update total and average loss
        total_loss += loss;
        avg_loss = (avg_loss + loss) / 2;

        // Compute predicted classes and update accuracy metrics
        predClasses = predictClassFromScores(scores);
        total_acc += computeAccuracy(predClasses, y_batch);
        avg_acc = total_acc / i;

        // Reset gradient of weights for the next iteration
        grad_w.fill(0.0);
        
        // Print loss and accuracy every 10 epochs
        if (i % 10 == 0){
            printf("Loss : %f | Accuracy : %f \n", avg_loss, avg_acc);
        }
    }
}
