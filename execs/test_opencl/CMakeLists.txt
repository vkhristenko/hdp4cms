message("compiling execs/test_opencl")

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
    add_executable(execs_test_opencl_${EXEC} ${SOURCE})
    message("hdp4cms libraries: ${HDP4CMS_LIBRARIES}")
    target_link_libraries(execs_test_opencl_${EXEC} ${ROOT_LIBRARIES} ${HDP4CMS_LIBRARIES} ${LINK_LIBRARIES})
    #    target_link_libraries(${EXEC} ${ROOT_LIBRARIES} ${HDP4CMS_LIBRARIES})
endforeach(SOURCE)
