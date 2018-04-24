# min version
cmake_minimum_required(VERSION 2.8.0)

# project 
project(analytics4cmssw)

# find openCL and add include directories
find_package(OpenCL REQUIRED)
if ( OpenCL_FOUND )
    include_directories(${OpenCL_INCLUDE_DIRS})
    message(OpenCL_FOUND = ${OpenCL_FOUND})
    message(OpenCL_INCLUDE_DIRS = ${OpenCL_INCLUDE_DIRS})
    message(OpenCL_LIBRARIES = ${OpenCL_LIBRARIES})
    message(OpenCL_VERSION_STRING = ${OpenCL_VERSION_MAJOR})
    message(OpenCL_VERSION_MAJOR = ${OpenCL_VERSION_MAJOR})
    message(OpenCL_VERSION_MINOR = ${OpenCL_VERSION_MINOR})
endif()

# for debug purposes
message(CMAKE_SOURCE_DIR = ${CMAKE_SOURCE_DIR})
message(CMAKE_BINARY_DIR = ${CMAKE_BINARY_DIR})
message(PROJECT_SOURCE_DIR = ${PROJECT_SOURCE_DIR})
message(PROJECT_BINARY_DIR = ${PROJECT_BINARY_DIR})

# add c++17 flag
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --std=c++17 -Wno-c++17-extensions")

# always follow the source tree for inclusions ( a la cmssw)
include_directories(${PROJECT_SOURCE_DIR})

# put eigen into the include path
message (EIGEN_HOME = ${EIGEN_HOME})
include_directories(${EIGEN_HOME})

# each subdir knows what to build
add_subdirectory(dataformats)
add_subdirectory(raw2digi)

# executables for various purposes
add_subdirectory(execs)
