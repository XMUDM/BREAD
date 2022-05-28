project(bread)
# Specify the minimum version for CMake
cmake_minimum_required(VERSION 2.8)

#SET(CMAKE_CXX_FLAGS "-std=c++11 -O3 -Wall")
SET(CMAKE_CXX_FLAGS "-std=c++11 -O3 -g -w -pthread -fopenmp -L/data/zqxu/fora/boost/lib -lboost_serialization")
#SET(CMAKE_CXX_FLAGS "-std=c++11 -O3 -w -pthread")


#find_package(Boost REQUIRED  unit_test_framework filesystem system date_time program_options serialization)
#find_package(Threads)

#set(HEADER algo.h build_oracle.h config.h graph.h head.h heap.h mylib.h query.h)
#set(PPRLIB  mylib.cpp config.cpp)

# Set the output folder where your program will be created
set(CMAKE_BINARY_DIR ${CMAKE_SOURCE_DIR})
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR})
set(LIBRARY_OUTPUT_PATH ${CMAKE_BINARY_DIR})

# The following folder will be included
include_directories("${PROJECT_SOURCE_DIR}")



add_executable(bread ${PROJECT_SOURCE_DIR}/bread.cpp ${PROJECT_SOURCE_DIR}/mylib.cpp ${PROJECT_SOURCE_DIR}/config.cpp)

target_link_libraries(bread pthread boost_serialization)
