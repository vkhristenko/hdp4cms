#include <iostream>

#include "raw2digi/pixel/interface/unpack.hpp"
#include "raw2digi/common/interface/dump_raw.hpp"
#include "dataformats/raw_fed/interface/fed_header.hpp"

namespace raw2digi::pixel {

using namespace dataformats::raw_fed;
using namespace dataformats::raw_fed;

void unpack(raw_buffer const& buffer, int fed) {
    /*
    PRINT(buffer.size());

    // get the data
    unsigned char const* data = &buffer[0];

    // extract the fed header
    fed_header const* fedheader = reinterpret_cast<fed_header const*>(data);
    PRINT(fedheader->fedid());
    PRINT(fedheader->bx());
    PRINT(fedheader->l1a());
    PRINT(fedheader->triggertype());
    PRINT(fedheader->boe());

    int nwords64 = buffer.size() / sizeof(uint64_t);
    int ndatawords32 = 2*(nwords64 - 2);
    uint64_t const *trailer = reinterpret_cast<uint64_t const*>(fedheader) + nwords64 - 1;
    uint32_t const *first_word = reinterpret_cast<uint32_t const*>(fedheader+1);
    uint32_t const *last_word = reinterpret_cast<uint32_t const*>(trailer-1)+1;

    for (auto word=first_word; word<=last_word; word++) {
        int link = ((*word) >> 26) & 0x3F;
        int roc = ((*word) >> 21) & 0x1F;
        int dcol = ((*word) >> 16) & 0x1F;
        int pixel = ((*word) >> 8) & 0xFF;
        int adc = (*word) & 0xFF;

        PRINT(link);
        PRINT(roc);
        PRINT(dcol);
        PRINT(pixel);
        PRINT(adc);
    }

    // dump the whole buffer
    printf("\n\n***********************************\n");
    printf("    FED=%d Dumping the whole RAW Buffer size = %lu Bytes\n", fed, buffer.size());
    printf("***********************************\n\n");
    raw2digi::common::dump_raw(buffer, 8);
    */
}

}
