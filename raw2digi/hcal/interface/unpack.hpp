#ifndef raw2digi_hcal_unpack_h
#define raw2digi_hcal_unpack_h

#include "dataformats/raw_fed/interface/raw_buffer.hpp"

namespace raw2digi::hcal {

void unpack(dataformats::raw_fed::raw_buffer const& buffer, int fed);

}

#endif
