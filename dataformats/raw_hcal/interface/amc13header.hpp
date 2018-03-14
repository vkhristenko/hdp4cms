#ifndef dataformats_raw_hcal_amc13header_h
#define dataformats_raw_hcal_amc13header_h

#include "dataformats/raw_hcal/interface/module_header.hpp"

namespace dataformats::raw_hcal {

class amc13header {
public:
    inline int namc() const { return int((amc13_header >> 52) & 0x0F); }
    inline int amc13formatversion() const { return int((amc13_header >> 60) & 0xF); }
    inline int sourceid() const { return int((cdf_header >> 8) & 0xFFF); }
    inline int bunchid() const { return int((cdf_header >> 20) & 0xFFF); }
    inline int l1anumber() const { return int((cdf_header >> 32) & 0x00FFFFFF); }
    inline int cdfeventtype() const { return int((cdf_header >> 56) & 0x0F); }
    inline unsigned int orbitnumber() const { return (unsigned int)((amc13_header >> 4) & 0xFFFFFFFFu); }

    inline module_header const& getm(int i) const { return modules_headers[i]; }
    uint64_t const* payload(int imod) const {
        uint64_t const *ptr = (&cdf_header) + 2 + namc();
        for (auto i=0; i<imod; i++)
            ptr += modules_headers[i].amcsize();
        return ptr;
    }

private:
    uint64_t cdf_header; // fed_header
    uint64_t amc13_header;
    module_header modules_headers[12];
};

}

#endif
