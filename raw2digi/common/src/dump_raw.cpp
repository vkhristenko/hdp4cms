#include "raw2digi/common/interface/dump_raw.hpp"

namespace raw2digi::common {

void dump_raw(dataformats::raw_fed::raw_buffer const& buffer, int countermax) {
    auto counter = 0;
    for (auto it = buffer.begin(); it!=buffer.end(); counter++, ++it) {
        if (counter % countermax == 0)
            printf("\n");
        printf("%02x ", *it);
    }
    printf("\n\n");
}

}
