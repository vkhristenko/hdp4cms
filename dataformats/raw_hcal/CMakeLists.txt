# find ROOT
find_package(ROOT REQUIRED COMPONENTS RHTTP)

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

# find all the sources
file(GLOB SOURCES src/*.cpp)

# generate shared libs
add_library(dataformats__fed_hcal SHARED ${SOURCES})
target_link_libraries(dataformats__fed_hcal ${ROOT_LIBRARIES})
