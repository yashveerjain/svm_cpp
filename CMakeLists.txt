
cmake_minimum_required(VERSION 3.10)
project(SVM_CPP)

set(CMAKE_CXX_STANDARD 17)

find_package(OpenCV 4 REQUIRED)
find_package(Eigen3 3.4 REQUIRED)
# If Eigen is not found using default search paths
# if(NOT TARGET Eigen3::Eigen)
  # Set the path to your custom Eigen include directory
# set(Eigen3 "/mnt/c/Users/yashv/Documents/projects/cpp_packages/eigen-3.4.0/Eigen")
  # Add the custom include directory to the search path
#   include_directories(${EIGEN_INCLUDE_DI})
# endif()

include_directories(include/ ${OpenCV_INCLUDE_DIRS})

include(CMakePrintHelpers)
cmake_print_variables(Eigen3)
cmake_print_variables(OpenCV_LIBS)
cmake_print_variables(EIGEN_INCLUDE_DIR)

add_library(svm_lib src/svm.cpp src/data_utils.cpp)

add_executable(test src/test.cpp)
target_link_libraries(test svm_lib ${OpenCV_LIBS})
# target_link_libraries()