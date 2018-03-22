#ifndef dataformats_raw_hcal_digi_collection_h
#define dataformats_raw_hcal_digi_collection_h

#include <vector>

#include "dataformats/raw_hcal/interface/channel_data.hpp"

namespace dataformats::raw_hcal {

template<typename T>
class digi_collection {
public:
    digi_collection() :
        m_nsamples(-1)
    {}
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
    {}

    digi_collection& operator=(digi_collection const& rhs) {
        m_nsamples = rhs.m_nsamples;
        m_data = rhs.m_data;
        return *this;
    }
    digi_collection& operator=(digi_collection&& rhs) {
        m_nsamples = std::move(rhs.m_nsamples);
        m_data = std::move(rhs.m_data);
        return *this;
    }

    // get the number of digis in a collection
    inline int size() const { 
        return m_data.size()==0 ? 0 : 
            m_data.size() / (T::HEADER_WORDS + m_nsamples*T::WORDS_PER_SAMPLE);
    }

    // get the raw data
    std::vector<uint16_t>& data() { return m_data; }

    // push an element
    void push_back(T const& digi) {
        uint16_t const* data = digi.get_data();

        // push header words
        for (auto ih=0; ih < T::HEADER_WORDS; ih++, data++)
            m_data.push_back(*data);

        // do this only once
        if (m_nsamples==-1) 
            m_nsamples = digi.nsamples();

        // push sample words
        for (auto is=0; is<digi.nsamples()*T::WORDS_PER_SAMPLE; is++, data++)
            m_data.push_back(*data);

    }

    T at(int i) { return operator[](i); } 
    T operator[](int i) {
        int ipos = i*(T::HEADER_WORDS + m_nsamples*T::WORDS_PER_SAMPLE);
        uint16_t const* data_start = m_data.data() + ipos;
        return T(data_start, m_nsamples);
    }

    // set the nsamples
    void set_nsamples(int nsamples) { m_nsamples = nsamples; }
    // get the nsamples
    int get_nsamples() const { return m_nsamples; }

private:
    int                                     m_nsamples;        
    std::vector<uint16_t>                   m_data;
};

using digi_collection_f01 = digi_collection<data_f01>;
using digi_collection_f2 = digi_collection<data_f2>;
using digi_collection_f3 = digi_collection<data_f3>;
using digi_collection_f4 = digi_collection<data_f4>;
using digi_collection_f5 = digi_collection<data_f5>;
using collections = std::tuple<digi_collection_f01, 
                               digi_collection_f2,
                               digi_collection_f3, 
                               digi_collection_f4,
                               digi_collection_f5>;

}

#endif
