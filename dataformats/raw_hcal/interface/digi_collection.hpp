#ifndef dataformats_raw_hcal_digi_collection_h
#define dataformats_raw_hcal_digi_collection_h

#include <vector>

#include "dataformats/raw_hcal/interface/channel_data.hpp"

namespace dataformats::raw_hcal {

template<typename T>
class digi_collection {
public:
    digi_collection(int nsamples) :
        m_nsamples(nsamples)
    {}
    // copy ctor
    digi_collection(digi_collection const& rhs) :
        m_nsamples(rhs.m_nsamples), m_data(rhs.m_data)
    {}
    // move ctor
    digi_collection(digi_collection&& rhs) :
        m_nsamples(rhs.m_nsamples), m_data(std::move(rhs.m_data))

    void push_back(T const&);
    T& at(int i) = operator[](i);
    T& operator[](int i);
private:
    int                                     m_nsamples;        
    std::vector<uint16_t>                   m_data;
};

template<data_f01>
digi_collection<data_f01> {
};

}

#endif
