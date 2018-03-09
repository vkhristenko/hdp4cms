# min version
cmake_minimum_required(VERSION 2.8.0)

# project 
project(analytics4cmssw)

# for debug purposes
message(CMAKE_SOURCE_DIR = ${CMAKE_SOURCE_DIR})
message(CMAKE_BINARY_DIR = ${CMAKE_BINARY_DIR})
message(PROJECT_SOURCE_DIR = ${PROJECT_SOURCE_DIR})
message(PROJECT_BINARY_DIR = ${PROJECT_BINARY_DIR})

# each subdir knows what to build
add_subdirectory(dataproc)
