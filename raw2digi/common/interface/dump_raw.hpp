#ifndef raw2digi_common_dump_raw_h
#define raw2digi_common_dump_raw_h

#include "dataformats/raw_fed/interface/raw_buffer.hpp"

namespace raw2digi::common {

#define PRINT(x) \
    std::cout << #x " = " << x  << std::endl

#define PRINTRAW(x) \
    printf(#x " = %02x\n", x);

void dump_raw(dataformats::raw_fed::raw_buffer const&, int countermax = 25);

}

#endif
