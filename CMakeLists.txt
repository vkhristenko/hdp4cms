# min version
cmake_minimum_required(VERSION 2.8.0)

# project 
project(analytics4cmssw)

# for debug purposes
message(CMAKE_SOURCE_DIR = ${CMAKE_SOURCE_DIR})
message(CMAKE_BINARY_DIR = ${CMAKE_BINARY_DIR})
message(PROJECT_SOURCE_DIR = ${PROJECT_SOURCE_DIR})
message(PROJECT_BINARY_DIR = ${PROJECT_BINARY_DIR})

# add c++17 flag
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++1z -Wno-c++1z-extensions")

# always follow the source tree for inclusions ( a la cmssw)
include_directories(${PROJECT_SOURCE_DIR})

# each subdir knows what to build
add_subdirectory(dataproc)
add_subdirectory(dataformats)
add_subdirectory(raw2digi)
