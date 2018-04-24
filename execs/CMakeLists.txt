cmake_minimum_required(VERSION 2.8.0)

message("execs/")
message(CMAKE_SOURCE_DIR = ${CMAKE_SOURCE_DIR})
message(CMAKE_BINARY_DIR = ${CMAKE_BINARY_DIR})
message(PROJECT_SOURCE_DIR = ${PROJECT_SOURCE_DIR})
message(PROJECT_BINARY_DIR = ${PROJECT_BINARY_DIR})

add_subdirectory(raw2digi)
add_subdirectory(test_opencl)
add_subdirectory(test_eigen)
