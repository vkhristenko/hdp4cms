#ifndef dataformats_raw_hcal_module_header_h
#define dataformats_raw_hcal_module_header_h

#include <cstdint>

namespace dataformats::raw_hcal {

class module_header {
public:
    inline uint16_t amcid() const { return uint16_t(header & 0xFFFF); }
    inline int amcslot() const { return int((header >> 16) & 0xF); }
    inline int amcblocknumber() const { return int((header >> 20) & 0xFF); }
    inline int amcsize() const { return int((header >> 32) & 0xFFFFFF); }
    inline bool amcmore() const { return ((header >> 61) & 0x1) != 0; }
    inline bool amcsegmented() const { return ((header >> 60) & 0x1) != 0; }
    inline bool amclengthok() const { return ((header >> 62) & 0x1) != 0; }
    inline bool amccrok() const { return ((header  >> 56) & 0x1) != 0 ;}
    inline bool amcdatapresent() const { return ((header >> 58) & 0x1)!=0; }
    inline bool amcdatavalid() const { return ((header >> 57) & 0x1)!=0; }
    inline bool amcenabled() const { return ((header >> 59) & 0x1)!=0; }

private:
    uint64_t header;
};

}

#endif
