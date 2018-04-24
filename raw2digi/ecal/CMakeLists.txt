message("compiling raw2digi/ecal")

# add root defs / include dirs 
add_definitions(${ROOT_CXX_FLAGS})
include_directories("${ROOT_INCLUDE_DIRS}")

# find all the sources
file(GLOB SOURCES src/*.cpp)

add_library(raw2digi__ecal SHARED ${SOURCES})
target_link_libraries(raw2digi__ecal ${ROOT_LIBRARIES} raw2digi__common)
