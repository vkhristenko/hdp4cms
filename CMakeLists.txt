# min version
cmake_minimum_required(VERSION 3.8.0)

set(CMAKE_VERBOSE_MAKEFILE OFF)

# project 
project(analytics4cmssw)

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/modules ${CMAKE_MODULE_PATH})

# for debug purposes
message(CMAKE_MODULE_PATH = ${CMAKE_MODULE_PATH})
message(CMAKE_PREFIX_PATH = ${CMAKE_PREFIX_PATH})
message(CMAKE_SOURCE_DIR = ${CMAKE_SOURCE_DIR})
message(CMAKE_BINARY_DIR = ${CMAKE_BINARY_DIR})
message(PROJECT_SOURCE_DIR = ${PROJECT_SOURCE_DIR})
message(PROJECT_BINARY_DIR = ${PROJECT_BINARY_DIR})
message (CMAKE_CXX_COMPILER_VERSION = ${CMAKE_CXX_COMPILER_VERSION})
message (CMAKE_CXX_COMPILER_ID = ${CMAKE_CXX_COMPILER_ID})

message("-------------------------------------------")

# find stuff based on the os
if (WIN32)
    message ("we are on win32")
endif (WIN32)
if (UNIX AND NOT APPLE)
    message ("we are on linux")

    # cuda
    if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
        if (CMAKE_CXX_COMPILER_VERSION VERSION_LESS "6.0.0")
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
            find_package(NVVM)
            message(NVVM_LIB = ${NVVM_LIB})
        endif()
    elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        if (CMAKE_CXX_COMPILER_VERSION VERSION_LESS "7.0.0")
            find_package(CUDA)
            if ( CUDA_FOUND )
                add_definitions(-DUSE_CUDA)
                message(CUDA_FOUND = ${CUDA_FOUND})
                message(NVVM_LIB = ${NVVM_LIB})
                message(CUDA_LIBRARIES = ${CUDA_LIBRARIES})
                message(CUDA_INCLUDE_DIRS = ${CUDA_INCLUDE_DIRS})
                #        set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++11" )
                include_directories(${CUDA_INCLUDE_DIRS} SYSTEM)
                cuda_include_directories(${CUDA_INCLUDE_DIRS})
            endif(CUDA_FOUND)
            find_package(NVVM)
            message(NVVM_LIB = ${NVVM_LIB})
        endif()
    endif()
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
    if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
        if (CMAKE_CXX_COMPILER_VERSION VERSION_LESS "6.0.0")
            find_package(CUDA)
            if ( CUDA_FOUND )
                add_definitions(-DUSE_CUDA)
                message(CUDA_FOUND = ${CUDA_FOUND})
                message(NVVM_LIB = ${NVVM_LIB})
                message(CUDA_LIBRARIES = ${CUDA_LIBRARIES})
                message(CUDA_INCLUDE_DIRS = ${CUDA_INCLUDE_DIRS})
                include_directories(${CUDA_INCLUDE_DIRS})

            endif(CUDA_FOUND)
            find_package(NVVM)
            message(NVVM_LIB = ${NVVM_LIB})
        endif()
    elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        if (CMAKE_CXX_COMPILER_VERSION VERSION_LESS "7.0.0")
            find_package(CUDA)
            if ( CUDA_FOUND )
                add_definitions(-DUSE_CUDA)
                message(CUDA_FOUND = ${CUDA_FOUND})
                message(NVVM_LIB = ${NVVM_LIB})
                message(CUDA_LIBRARIES = ${CUDA_LIBRARIES})
                message(CUDA_INCLUDE_DIRS = ${CUDA_INCLUDE_DIRS})
                #        set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++11" )
                include_directories(${CUDA_INCLUDE_DIRS} SYSTEM)
                cuda_include_directories(${CUDA_INCLUDE_DIRS})
            endif(CUDA_FOUND)
            find_package(NVVM)
            message(NVVM_LIB = ${NVVM_LIB})
        endif()
    endif()
endif(APPLE)

message("-------------------------------------------")

find_package(ArrayFire)
if (ArrayFire_FOUND)
    message("ArrayFire is found!")
    message(ArrayFire_CPU_FOUND = ${ArrayFire_CPU_FOUND})
    message(ArrayFire_CPU_LIBRARIES = ${ArrayFire_CPU_LIBRARIES})
    message(ArrayFire_CUDA_FOUND = ${ArrayFire_CUDA_FOUND})
    message(ArrayFire_CUDA_LIBRARIES = ${ArrayFire_CUDA_LIBRARIES})
    message(ArrayFire_OpenCL_FOUND = ${ArrayFire_OpenCL_FOUND})
    message(ArrayFire_OpenCL_LIBRARIES = ${ArrayFire_OpenCL_LIBRARIES})
    message(ArrayFire_Unified_FOUND = ${ArrayFire_Unified_FOUND})
    message(ArrayFire_Unified_LIBRARIES = ${ArrayFire_Unified_LIBRARIES})

    INCLUDE_DIRECTORIES(${ArrayFire_INCLUDE_DIRS})
else()
    message("ArrayFire is not found!")
endif()

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
if (CUDA_FOUND)
    cuda_include_directories(${EIGEN_HOME})
endif()

# each subdir knows what to build
add_subdirectory(dataformats)
add_subdirectory(raw2digi)

# executables for various purposes
add_subdirectory(execs)
