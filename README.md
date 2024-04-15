

# SVM for cpp


## Dataset used
* Install Cifar10 dataset which have the files in the `.bin` format. Link to [dataset](https://www.cs.toronto.edu/~kriz/cifar-100-binary.tar.gz).
* Unzip the file and extract in this directory.

## Linear SVM
* Instruction to install the package and run the file
'''
mkdir -p build && cd build
cmake ..
make
./test ../data/cifar-10-batches-bin/data_batch_1.bin 1000 0.01 100  
'''

* Details of the command : ./test `dataset_filepath` `batch size` `learning rate` `epochs`
* Supports regularization but not in argument right now.
* can show the image of the extracted data.

## Requirements
1. Eigen>=3.4 (http://eigen.tuxfamily.org/index.php?title=Main_Page#Download)
    - unzip the file, and run the following command:
    '''
    cd eigen-3.4.0
    mkdir build && cd build
    cmake ..
    sudo make install
    '''
2. Opencv>=4

## Issues
* Training is not converging some issues with training pipeline. (Need to debug).