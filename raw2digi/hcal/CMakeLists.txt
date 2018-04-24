message("compiling raw2digi/hcal")

# add root defs / include dirs 
add_definitions(${ROOT_CXX_FLAGS})
include_directories("${ROOT_INCLUDE_DIRS}")

# find all the sources
file(GLOB SOURCES src/*.cpp)

add_library(raw2digi__hcal SHARED ${SOURCES})
target_link_libraries(raw2digi__hcal ${ROOT_LIBRARIES} raw2digi__common)
