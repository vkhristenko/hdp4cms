#ifndef raw2digi_common_fed_numbering_h
#define raw2digi_common_fed_numbering_h

namespace raw2digi::common {

#define MINECALFEDID 600
#define MAXECALFEDID 670
#define MINHCALuTCAFEDID 1100
#define MAXHCALuTCAFEDID 1199
#define MINSiPixelFEDID 0
#define MAXSiPixelFEDID 40
#define MINSiPixeluTCAFEDID 1200
#define MAXSiPixeluTCAFEDID 1349
#define MINSiPixel2nduTCAFEDID 1500
#define MAXSiPixel2nduTCAFEDID 1649

bool is_hcal_fed(int fed);

bool is_ecal_fed(int fed);

bool is_pixel_fed(int fed);

}

#endif
