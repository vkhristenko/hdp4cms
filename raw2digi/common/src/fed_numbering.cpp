#include "raw2digi/common/interface/fed_numbering.hpp"

namespace raw2digi::common {

bool is_hcal_fed(int fed) {
    return fed>=MINHCALuTCAFEDID && fed<=MAXHCALuTCAFEDID;
}

bool is_ecal_fed(int fed) {
    return fed>=MINECALFEDID && fed<=MAXECALFEDID;
}

}
