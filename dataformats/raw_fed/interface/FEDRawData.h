#ifndef dataformtats_raw_fed_FEDRawData_h
#define dataformtats_raw_fed_FEDRawData_h

#include <vector>
#include <cstddef>
#include "TObject.h"

class FEDRawData {
public:
    std::vector<unsigned char> data_;
    FEDRawData() {}
    FEDRawData(const FEDRawData & rhs) :
        data_(rhs.data_) 
    {}
    virtual ~FEDRawData() {}
   
    // for io
    ClassDef(FEDRawData, 12); // Generated by MakeProject.
};

ClassImp(FEDRawData)

#endif
