# min version
cmake_minimum_required(VERSION 3.8.0)

set(CMAKE_VERBOSE_MAKEFILE OFF)

# project 
project(analytics4cmssw)

# for debug purposes
message(CMAKE_MODULE_PATH = ${CMAKE_MODULE_PATH})
message(CMAKE_PREFIX_PATH = ${CMAKE_PREFIX_PATH})
message(CMAKE_SOURCE_DIR = ${CMAKE_SOURCE_DIR})
message(CMAKE_BINARY_DIR = ${CMAKE_BINARY_DIR})
message(PROJECT_SOURCE_DIR = ${PROJECT_SOURCE_DIR})
message(PROJECT_BINARY_DIR = ${PROJECT_BINARY_DIR})

message("-------------------------------------------")

# find stuff based on the os
if (WIN32)
    message ("we are on win32")
endif (WIN32)
if (UNIX AND NOT APPLE)
    message ("we are on linux")

    # cuda
    find_package(CUDA)
    if ( CUDA_FOUND )
        add_definitions(-DUSE_CUDA)
        message(CUDA_FOUND = ${CUDA_FOUND})
        message(CUDA_LIBRARIES = ${CUDA_LIBRARIES})
        message(CUDA_INCLUDE_DIRS = ${CUDA_INCLUDE_DIRS})
        #        set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++11" )
        include_directories(${CUDA_INCLUDE_DIRS} SYSTEM)
        cuda_include_directories(${CUDA_INCLUDE_DIRS})
    endif(CUDA_FOUND)
endif()
if (APPLE)
    message ("we are on mac")

    # opencl
    find_package(OpenCL)
    if ( OpenCL_FOUND )
        include_directories(${OpenCL_INCLUDE_DIRS})
        message(OpenCL_FOUND = ${OpenCL_FOUND})
        message(OpenCL_INCLUDE_DIRS = ${OpenCL_INCLUDE_DIRS})
        message(OpenCL_LIBRARIES = ${OpenCL_LIBRARIES})
        message(OpenCL_VERSION_STRING = ${OpenCL_VERSION_MAJOR})
        message(OpenCL_VERSION_MAJOR = ${OpenCL_VERSION_MAJOR})
        message(OpenCL_VERSION_MINOR = ${OpenCL_VERSION_MINOR})

        # to turn flags on to use opencl stuff
        add_definitions(-DUSE_OPENCL)
    endif(OpenCL_FOUND)

    # cuda
    find_package(CUDA)
    if ( CUDA_FOUND )
        add_definitions(-DUSE_CUDA)
        message(CUDA_FOUND = ${CUDA_FOUND})
        message(CUDA_LIBRARIES = ${CUDA_LIBRARIES})
        message(CUDA_INCLUDE_DIRS = ${CUDA_INCLUDE_DIRS})
        include_directories(${CUDA_INCLUDE_DIRS})

    endif(CUDA_FOUND)
endif(APPLE)

message("-------------------------------------------")

# find ROOT
find_package(ROOT REQUIRED)
message(CMAKE_MODULE_PATH = ${CMAKE_MODULE_PATH})
message(CMAKE_PREFIX_PATH = ${CMAKE_PREFIX_PATH})

# check if ROOT is/where found
message(ROOT_FOUND = ${ROOT_FOUND})
message(ROOT_INCLUDE_DIRS = ${ROOT_INCLUDE_DIRS})
message(ROOT_LIBRARIES = ${ROOT_LIBRARIES})
message(ROOT_DEFINITIONS = ${ROOT_DEFINITIONS})
message(ROOT_CXX_FLAGS = ${ROOT_CXX_FLAGS})
message(ROOT_CC_FLAGS = ${ROOT_CC_FLAGS})

# add root defs / include dirs 
add_definitions(${ROOT_CXX_FLAGS})
include_directories("${ROOT_INCLUDE_DIRS}")

# add c++17 flag
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --std=c++17 -Wno-c++17-extensions")

message("-------------------------------------------")

# always follow the source tree for inclusions ( a la cmssw)
include_directories(${PROJECT_SOURCE_DIR})

# put eigen into the include path
message (EIGEN_HOME = ${EIGEN_HOME})
include_directories(${EIGEN_HOME} SYSTEM)

# each subdir knows what to build
add_subdirectory(dataformats)
add_subdirectory(raw2digi)

# executables for various purposes
add_subdirectory(execs)
