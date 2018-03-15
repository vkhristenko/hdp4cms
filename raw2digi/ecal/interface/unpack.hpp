#ifndef raw2digi_ecal_unpack_h
#define raw2digi_ecal_unpack_h

#include "dataformats/raw_fed/interface/raw_buffer.hpp"

namespace raw2digi::ecal {

void unpack(dataformats::raw_fed::raw_buffer const&, int fed);

}

#endif
