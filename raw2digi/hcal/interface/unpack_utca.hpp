#ifndef raw2digi_hcal_unpack_utca_h
#define raw2digi_hcal_unpack_utca_h

#include "dataformats/raw_fed/interface/raw_buffer.hpp"
#include "dataformats/raw_hcal/interface/digi_collection.hpp"
#include <tuple>

namespace raw2digi::hcal {

using namespace dataformats::raw_hcal;

std::tuple<digi_collection_f01, 
           digi_collection_f2, 
           digi_collection_f3,
           digi_collection_f4,
           digi_collection_f5> 
unpack_utca(dataformats::raw_fed::raw_buffer const& buffer);

}

#endif
