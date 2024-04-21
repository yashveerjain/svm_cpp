
cmake_minimum_required(VERSION 3.10)
project(SVM_CPP)

set(CMAKE_CXX_STANDARD 17)

# Fo threading in cpp need to set pthread
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")

find_package(OpenCV 4 REQUIRED)
find_package(Eigen3 3.4 REQUIRED)


include_directories(include/ ${OpenCV_INCLUDE_DIRS})

include(CMakePrintHelpers)
cmake_print_variables(Eigen3)
cmake_print_variables(OpenCV_LIBS)
cmake_print_variables(EIGEN_INCLUDE_DIR)

add_library(svm_lib src/svm.cpp src/data_utils.cpp)

add_executable(test src/test.cpp)
add_executable(train src/train.cpp)
target_link_libraries(test svm_lib ${OpenCV_LIBS})
target_link_libraries(train svm_lib ${OpenCV_LIBS})
# target_link_libraries()