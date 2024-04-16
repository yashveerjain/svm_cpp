#pragma once

#include <iostream>
#include <vector>
#include <tuple>
#include <eigen3/Eigen/Core>

#include "data_utils.hpp"

// using namespace Eigen::placeholders;

namespace custom_ai {
    /**
     * @brief Class for Linear Support Vector Machine (Linear SVM)
     */
    class LinearSVM {
        custom_ai::MatrixXd_f _w; // Weight matrix
        custom_ai::MatrixXd_f _x; // Input feature matrix
        custom_ai::VectorXd_i _y; // Target vector
        int _feature_size; // Size of the input feature

    public:
        /**
         * @brief Construct a new Linear SVM object with specified feature size, output size, input data, and target labels
         * 
         * @param feature_size The size of input feature
         * @param output_size The size of output
         * @param input The input data matrix
         * @param target The target labels vector
         */
        LinearSVM(int feature_size, int output_size, custom_ai::MatrixXd_f& input, custom_ai::VectorXd_i& target);

        /**
         * @brief Default constructor for Linear SVM
         */
        LinearSVM(): _x(custom_ai::MatrixXd_f::Random(1000, 100)), // Default random input data matrix (1000x100)
                    _y(custom_ai::VectorXd_i::Zero(1000, 1)),       // Default target labels vector (1000x1)
                    _w(custom_ai::MatrixXd_f::Random(100, 10)) {}   // Default random weight matrix (100x10)

        /**
         * @brief Preprocess the input data
         * 
         * @param input The input data matrix
         * @param rows Number of rows in the input data
         * @param cols Number of columns in the input data
         * @return custom_ai::MatrixXd_f Preprocessed input data matrix
         */
        custom_ai::MatrixXd_f preprocessor(custom_ai::MatrixXd_f& input, const int& rows, const int& cols);

        /**
         * @brief Train the Linear SVM model
         * 
         * @param epoch Number of epochs (iterations) for training (default: 100)
         * @param lr Learning rate (default: 0.01)
         * @param batch_size Batch size for training (default: 32)
         * @param reg Regularization parameter (default: 0)
         */
        void train(int epoch = 100, float lr = 0.01, int batch_size = 32, float reg = 0);

        /**
         * @brief Predict class labels from scores predicted by the linear model
         * 
         * @param score Predicted scores for each class by the linear model
         * @return custom_ai::VectorXd_i Predicted class labels
         */
        custom_ai::VectorXd_i predictClassFromScores(custom_ai::MatrixXd_f& score);

        /**
         * @brief Perform inference using the trained Linear SVM model
         * 
         * @param test_data The input test data matrix
         * @return custom_ai::VectorXd_i Predicted class labels for the test data
         */
        custom_ai::VectorXd_i inference(custom_ai::MatrixXd_f& test_data);

        /**
         * @brief Compute the SVM loss
         * 
         * @param grad_w Gradient of weights
         * @param x Feature input
         * @param pred Predicted scores by the linear model
         * @param target Correct labels for the feature input
         * @return float Computed loss
         */
        float loss(custom_ai::MatrixXd_f& grad_w, const custom_ai::MatrixXd_f& x, const custom_ai::MatrixXd_f& pred, const custom_ai::VectorXd_i& target);

        /**
         * @brief Compute the accuracy of predictions
         * 
         * @param pred Predicted class labels
         * @param target Correct class labels
         * @return float Accuracy
         */
        float computeAccuracy(custom_ai::VectorXd_i pred, custom_ai::VectorXd_i target);

        /**
         * @brief Compute the SVM loss (vectorized version)
         * 
         * @param grad_w Gradient of weights
         * @param x Feature input
         * @param pred Predicted scores by the linear model
         * @param target Correct labels for the feature input
         * @return float Computed loss
         */
        float loss_vectoriesed(custom_ai::MatrixXd_f& grad_w, const custom_ai::MatrixXd_f& x, const custom_ai::MatrixXd_f& pred, const custom_ai::VectorXd_i& target);
    };
} // namespace custom_ai
