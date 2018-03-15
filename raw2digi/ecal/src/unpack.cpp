#include <iostream>

#include "raw2digi/ecal/interface/unpack.hpp"
#include "raw2digi/common/interface/dump_raw.hpp"
#include "dataformats/raw_fed/interface/fed_header.hpp"

namespace raw2digi::ecal {

using namespace dataformats::raw_fed;

void unpack(raw_buffer const& buffer, int fed) {
    PRINT(buffer.size());

    // get the data
    unsigned char const* data = &buffer[0];

    // extract the fed header info
    fed_header const* fedheader = reinterpret_cast<fed_header const*>(data);
    PRINT(fedheader->fedid());
    PRINT(fedheader->bx());
    PRINT(fedheader->l1a());
    PRINT(fedheader->triggertype());
    PRINT(fedheader->boe());

    // dump the whole buffer
    printf("\n\n***********************************\n");
    printf("    FED=%d Dumping the whole RAW Buffer size = %lu Bytes\n", fed, buffer.size());
    printf("***********************************\n\n");
    raw2digi::common::dump_raw(buffer, 8);
}

}
