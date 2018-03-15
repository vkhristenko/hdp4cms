#ifndef dataformats_raw_fed_fed_header_h
#define dataformats_raw_fed_fed_header_h

#include <cstdint>

namespace dataformats::raw_fed {

class fed_header {
public:
    inline int fov() const { return ((m_data >> 4) & 0xF); }
    inline int fedid() const { return ((m_data >> 8) & 0xFFF); }
    inline int bx() const { return ((m_data >> 20) & 0xFFF); }
    inline int l1a() const { return ((m_data >> 32) & 0xFFFFFF); }
    inline int triggertype() const { return ((m_data >> 56) & 0xF); }
    inline int boe() const { return ((m_data >> 60) & 0xF); }

private:
    uint64_t m_data;

};

}

#endif
