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

# generate dictionaries
include(${ROOT_USE_FILE})
ROOT_GENERATE_DICTIONARY(
    G__DICT__TEST
    LINKDEF conf/LinkDef.h
)
add_library(DICTTEST SHARED G__DICT__TEST.cxx)
target_link_libraries(DICTTEST ${ROOT_LIBRARIES})

# add executables
add_executable(dump_raw bin/dump_raw.cpp)
target_link_libraries(dump_raw ${ROOT_LIBRARIES} DICTTEST)
