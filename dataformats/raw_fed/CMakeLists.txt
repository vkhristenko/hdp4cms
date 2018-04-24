message("compiling dataformats/raw_fed")

# generate dictionaries
include(${ROOT_USE_FILE})
ROOT_GENERATE_DICTIONARY(
    G__DICT__DATAFORMATS__RAW_FED dataformats/raw_fed/interface/FEDRawData.h dataformats/raw_fed/interface/FEDRawDataCollection.h
    dataformats/raw_fed/interface/edm__ViewTypeChecker.h dataformats/raw_fed/interface/edm__WrapperBase.h dataformats/raw_fed/interface/edm__Wrapper_FEDRawDataCollection_.h
    LINKDEF conf/LinkDef.h
)
add_library(dataformats__raw_fed SHARED G__DICT__DATAFORMATS__RAW_FED.cxx)
target_link_libraries(dataformats__raw_fed ${ROOT_LIBRARIES})
