message("compiling raw2digi/common")

# add root defs / include dirs 
add_definitions(${ROOT_CXX_FLAGS})
include_directories("${ROOT_INCLUDE_DIRS}")

# find all the source files
file(GLOB SOURCES src/*.cpp)

add_library(raw2digi__common SHARED ${SOURCES})
target_link_libraries(raw2digi__common ${ROOT_LIBRARIES})
