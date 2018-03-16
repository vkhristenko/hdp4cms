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

# set a list of internal libraries to link against
# TODO: we should be able to detect this automatically
set(HDP4CMS_LIBRARIES 
    dataformats__raw_fed dataformats__raw_hcal 
    raw2digi__hcal raw2digi__common
    raw2digi__ecal
    raw2digi__pixel
)

# src directory should contain only source files with main entry point
file (GLOB SOURCES src/*.cpp)
foreach (SOURCE ${SOURCES})
    string(REPLACE ".cpp" "" EXEC ${SOURCE})
    get_filename_component(EXEC ${EXEC} NAME)
    message("adding executable ${EXEC}")
    add_executable(${EXEC} ${SOURCE})
    message("hdp4cms libraries: ${HDP4CMS_LIBRARIES}")
    target_link_libraries(${EXEC} ${ROOT_LIBRARIES} ${HDP4CMS_LIBRARIES} ${OpenCL_LIBRARIES})
    #    target_link_libraries(${EXEC} ${ROOT_LIBRARIES} ${HDP4CMS_LIBRARIES})
endforeach(SOURCE)
