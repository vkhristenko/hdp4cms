message("compiling dataformats/raw_hcal")

# add root defs / include dirs 
add_definitions(${ROOT_CXX_FLAGS})
include_directories("${ROOT_INCLUDE_DIRS}")

# find all the sources
file(GLOB SOURCES src/*.cpp)

# generate shared libs
add_library(dataformats__raw_hcal SHARED ${SOURCES})
target_link_libraries(dataformats__raw_hcal ${ROOT_LIBRARIES})
