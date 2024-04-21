# SVM for cpp
Implemented Linear SVM training on Cifar10 dataset using c++.

## Dataset used
* Install Cifar10 dataset which have the files in the `.bin` format. Link to [dataset](https://www.cs.toronto.edu/~kriz/cifar-100-binary.tar.gz).
* Unzip the file and extract in this directory.

## Linear SVM
* Instruction to install the package and run the file to train the model

```
mkdir -p build && cd build
cmake ..
make
./train ../data/cifar-10-batches-bin/data_batch_1.bin model.bin 1000 0.01 100 0.00001 show  
```

* Instruction to install the package and run the file to test the model given the model file is already present (if not train the model it will automatically store the model file for you)

```
./test../data/cifar-10-batches-bin/data_batch_1.bin model.bin show  
```

* Details of the command : ./test `dataset_filepath` `model path` `show sample image`
* can show the image of the extracted data.

## Requirements
1. Eigen>=3.4 (http://eigen.tuxfamily.org/index.php?title=Main_Page#Download)
    - unzip the file, and run the following command:
    ```
    cd eigen-3.4.0
    mkdir build && cd build
    cmake ..
    sudo make install
    ```
2. Opencv>=4

## TODO
* Add inference script
* Optimise loss function