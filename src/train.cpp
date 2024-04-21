
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
        custom_ai::MatrixXd_f input_data;
        custom_ai::VectorXd_i label;
        std::tie(input_data, label) = dataset.read(argv[1]);
        printf("Dataset has Feature size %d and classes %d\n",dataset.getFeatureSize(),dataset.getTotalClasses());

        printf("Input Feature from Cifar10 with rows %ld and cols %ld\n", input_data.rows(), input_data.cols());
        printf("Input target from Cifar10 with rows %ld and cols %ld\n", label.rows(), label.cols());
        // std::cout << input(Eigen::in,2);
        printf("Value of feature at row %d and col %d, is : %f\n",test_row,2,input_data(test_row,2));
        printf("Value of target at row %d and col %d, is : %d\n",test_row,1,label(test_row));

        svm::LinearSVM linear_svm(dataset.getFeatureSize(),dataset.getTotalClasses(),input_data, label);
        int batch_size = 512, epoch=100;
        float lr = 0.01, reg=0.00001;
        std::string model_path = "model.bin"; // default path
        if (argc>=2) model_path = argv[2];
        if (argc>=3) batch_size = std::stoi(argv[3]);
        if (argc>=4) lr = std::stof(argv[4]);
        if (argc>=5) epoch=std::stoi(argv[5]);
        if (argc>=6) reg = std::stof(argv[6]);
        if (argc>=7) {
        // take 1 image from any of the row from the input data, as each row is the image of cifar dataset.
            cv::Mat image = dataset.getImageFromData(input_data.row(test_row));

            custom_ai::showImage(label(test_row), image);
        }

        printf("Batch Size %d, Epoch %d, regularization : %f and lr : %f\n",batch_size,epoch,reg, lr);
        linear_svm.train(epoch, lr, batch_size, reg, true);
        linear_svm.saveModel(model_path);

    }
    catch (std::exception& e){
        std::cerr << "Error : "<<e.what()<<std::endl;
    }

}
