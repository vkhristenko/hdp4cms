#ifndef raw2digi_common_fed_numbering_h
#define raw2digi_common_fed_numbering_h

namespace raw2digi::common {

#define MINECALFEDID 600
#define MAXECALFEDID 670
#define MINHCALuTCAFEDID 1100
#define MAXHCALuTCAFEDID 1199

bool is_hcal_fed(int fed);

bool is_ecal_fed(int fed);

}

#endif
