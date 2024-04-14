
cmake_minimum_required(VERSION 3.10)
project(SVM_CPP)

set(CMAKE_CXX_STANDARD 17)

find_package(OpenCV 4 REQUIRED)
find_package(Eigen3 REQUIRED)

include_directories(include/ ${OpenCV_INCLUDE_DIRS})

add_library(svm_lib src/svm.cpp)

add_executable(test src/test.cpp)
target_link_libraries(test svm_lib ${OpenCV_LIBS} ${Eigen3})
# target_link_libraries()