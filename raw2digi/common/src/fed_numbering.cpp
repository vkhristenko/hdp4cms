#include "raw2digi/common/interface/fed_numbering.hpp"

namespace raw2digi::common {

bool is_hcal_fed(int fed) {
    return fed>=MINHCALuTCAFEDID && fed<=MAXHCALuTCAFEDID;
}

bool is_ecal_fed(int fed) {
    return fed>=MINECALFEDID && fed<=MAXECALFEDID;
}

bool is_pixel_fed(int fed) {
    return (fed>=MINSiPixelFEDID && MAXSiPixelFEDID) || (fed>=MINSiPixeluTCAFEDID && fed<=MAXSiPixeluTCAFEDID) ||
           (fed>=MINSiPixel2nduTCAFEDID && fed<=MAXSiPixel2nduTCAFEDID);
}

}
