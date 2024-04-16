
#include<iostream>
#include<stdexcept>
#include<string>

#include<opencv2/opencv.hpp>
#include<eigen3/Eigen/Core>
#include<eigen3/Eigen/Dense>
#include <opencv2/highgui.hpp>

#include "data_utils.hpp"
#include "svm.hpp"

using std::cout;
using std::endl;

int main(int argc, char *argv[]){


    if (argc<1) {
        cout<<"Please provide the filepath of dataset";
        return 1;
    }
    custom_ai::Cifar10 dataset;
    try{
        int test_row = 10;
        custom_ai::MatrixXd_f test_data;
        custom_ai::VectorXd_i test_label;
        std::tie(test_data, test_label) = dataset.read(argv[1]);
        printf("Dataset has Feature size %d and classes %d\n",dataset.getFeatureSize(),dataset.getTotalClasses());

        printf("test Feature from Cifar10 with rows %ld and cols %ld\n", test_data.rows(), test_data.cols());
        printf("test target from Cifar10 with rows %ld and cols %ld\n", test_label.rows(), test_label.cols());
        // std::cout << test(Eigen::in,2);
        printf("Value of feature at row %d and col %d, is : %f\n",test_row,2,test_data(test_row,2));
        printf("Value of target at row %d and col %d, is : %d\n",test_row,1,test_label(test_row));

        custom_ai::LinearSVM svm(dataset.getFeatureSize(),dataset.getTotalClasses(),test_data, test_label);
        
        std::string model_path = "model.bin"; // default path
        if (argc>=1) model_path = argv[2];
        if (argc>=2) {
        // take 1 image from any of the row from the test data, as each row is the image of cifar dataset.
            cv::Mat image = dataset.getImageFromData(test_data.row(test_row));

            custom_ai::showImage(test_label(test_row), image);
        }
        cout<<"Model Path : "<<model_path<<endl;
        svm.load_model(model_path);
        custom_ai::VectorXd_i predictions =  svm.predict(test_data);
        std::cout << "accuracy : "<<svm.computeAccuracy(predictions, test_label)<<std::endl;
    }
    catch (std::exception& e){
        std::cerr << "Error : "<<e.what()<<std::endl;
    }

}