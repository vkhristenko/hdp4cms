message("compiling dataformats/raw_fed")

# add root defs / include dirs 
add_definitions(${ROOT_CXX_FLAGS})
include_directories("${ROOT_INCLUDE_DIRS}")

# generate dictionaries
include(${ROOT_USE_FILE})
ROOT_GENERATE_DICTIONARY(
    G__DICT__DATAFORMATS__RAW_FED interface/FEDRawData.h interface/FEDRawDataCollection.h
    interface/edm__ViewTypeChecker.h interface/edm__WrapperBase.h interface/edm__Wrapper_FEDRawDataCollection_.h
    LINKDEF conf/LinkDef.h
)
add_library(dataformats__raw_fed SHARED G__DICT__DATAFORMATS__RAW_FED.cxx)
target_link_libraries(dataformats__raw_fed ${ROOT_LIBRARIES})
