#ifndef raw2digi_pixel_unpack_h
#define raw2digi_pixel_unpack_h

#include "dataformats/raw_fed/interface/raw_buffer.hpp"

namespace raw2digi::pixel {

void unpack(dataformats::raw_fed::raw_buffer const&, int fed);

}

#endif
