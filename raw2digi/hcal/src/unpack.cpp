#include <iostream>

#include "raw2digi/hcal/interface/unpack.hpp"
#include "raw2digi/hcal/interface/unpack_utca.hpp"
#include "raw2digi/common/interface/dump_raw.hpp"

namespace raw2digi::hcal {

void unpack(dataformats::raw_fed::raw_buffer const& buffer, int fed) {
    PRINT(buffer.size());

    // pointer to the first byte of the buffer
    unsigned char const *data = &buffer[0];

    // BOEshouldBeZeroAlways() const { return ( (commondataformat3>>28) & 0x0F); }
    // /** Get the Beginning Of Event bits.  If it's not the first or last CDF Slink64 word, the high 4 bits must be zero.*/
    short boe = (data[12] >> 4) & 0x0F;
    PRINTRAW(data[12]);
    PRINTRAW(boe);

    if (boe == 0) {
        printf("vme fed found\n");
        return;
    }
    unpack_utca(buffer);

    // dump the whole buffer
    printf("\n\n***********************************\n");
    printf("    FED=%d Dumping the whole RAW Buffer size = %lu Bytes\n", fed, buffer.size());
    printf("***********************************\n\n");
    raw2digi::common::dump_raw(buffer, 8);
}

}
