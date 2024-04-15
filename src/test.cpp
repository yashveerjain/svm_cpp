
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

    /*
        Testing the Linear SVM Model
        // Create custom data to test the svm class training pipeline
        int feature_size = 100;
        int output_size = 10;
        int total_samples  = 1000;
        custom_ai::MatrixXd_f input = custom_ai::MatrixXd_f::Ones(total_samples,feature_size);
        custom_ai::VectorXd_i target = custom_ai::VectorXd_i::Random(total_samples,1);

        for (int i =0; i<total_samples;i++){
            
            target[i]= std::abs(target[i]%output_size);
            // cout<<i << " "<< target[i]<<endl;
            // target[i] = std::abs(target[i]);
        }
        // target = (target + custom_ai::VectorXd_i::Ones(total_samples,1))%(output_size);
        printf("Random Feature created with rows %ld and cols %ld\n", input.rows(), input.cols());
        printf("Random target created with rows %ld and cols %ld\n", target.rows(), target.cols());
        // std::cout << input(Eigen::in,2);
        printf("Value of feature at row %d and col %d, is : %f\n",10,2,input(10,2));
        printf("Value of target at row %d and col %d, is : %d\n",10,1,target(10));

        //testing linear svm loss and train method
        custom_ai::LinearSVM svm(feature_size,output_size,input, target);
        svm.train(11);


    */
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

        // take 1 image from any of the row from the input data, as each row is the image of cifar dataset.
        // cv::Mat image = dataset.getImageFromData(input_data.row(test_row));

        // custom_ai::showImage(label(test_row), image);

        custom_ai::LinearSVM svm(dataset.getFeatureSize(),dataset.getTotalClasses(),input_data, label);
        int batch_size = 512, epoch=100;
        float lr = 0.01;
        if (argc>=2) batch_size = std::stoi(argv[2]);
        if (argc>=3) lr = std::stof(argv[3]);
        if (argc>=4) epoch=std::stoi(argv[4]);

        printf("Batch Size %d, Epoch %d, and lr : %f\n",batch_size,epoch,lr);
        svm.train(epoch, lr, batch_size);
    }
    catch (std::exception& e){
        std::cerr << "Error : "<<e.what()<<std::endl;
    }

}