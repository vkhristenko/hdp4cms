#ifndef raw2digi_hcal_unpack_h
#define raw2digi_hcal_unpack_h

#include "dataformats/raw_fed/interface/raw_buffer.hpp"
#include "dataformats/raw_hcal/interface/digi_collection.hpp"

namespace raw2digi::hcal {

using namespace dataformats::raw_hcal;

std::tuple<digi_collection_f01,
           digi_collection_f2,
           digi_collection_f3,
           digi_collection_f4,
           digi_collection_f5>
unpack(dataformats::raw_fed::raw_buffer const& buffer, int fed);

}

#endif
