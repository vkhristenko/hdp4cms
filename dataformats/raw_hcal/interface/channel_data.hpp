#ifndef dataformats_raw_hcal_channel_data_h
#define dataformats_raw_hcal_channel_data_h

#include <cstdio>

namespace dataformats::raw_hcal {

class data_f01 {
public:
    // decls, defs
    static int constexpr WORDS_PER_SAMPLE = 1;
    static int constexpr HEADER_WORDS = 1;
    // TODO: what is this guy?
    static int constexpr FLAG_WORDS = 1;

    // constructor
    data_f01(uint16_t const* start, uint16_t const* end) :
        data(start), samples(int((end - start - HEADER_WORDS)/WORDS_PER_SAMPLE))
    {}
    ~data_f01()
    {}

    class sample {
    public:
        sample(uint16_t data) :
            m_data(data)
        {}

        inline bool soi() const { return ((m_data >> 14) & 0x1); }
        inline int tdc() const { return ((m_data >> 8) & 0x3F); }
        inline int adc() const { return (m_data & 0x00FF); }

    private:
        uint16_t m_data;
    };

    inline int nsamples() const { return samples; }
    inline int channelid() const { return ((*data) & 0xFF); }
    inline int capid() const { return (((*data) >> 8) & 0x3); }
    inline sample get_sample(int i) const {
        return sample(*(data + HEADER_WORDS + i*WORDS_PER_SAMPLE));
    }

private:
    uint16_t const *data;
    int samples;
};

class data_f2 {
public:
    // decls, defs
    static int constexpr WORDS_PER_SAMPLE = 2;
    static int constexpr HEADER_WORDS = 1;
    // TODO: what is this guy?
    static int constexpr FLAG_WORDS = 1;

    data_f2(uint16_t const* start, uint16_t const *end) :
        data(start), samples(int((end - start - HEADER_WORDS)/WORDS_PER_SAMPLE))
    {}
    ~data_f2()
    {}

    class sample {
    public:
        sample(uint16_t word1, uint16_t word2) :
            m_word1(word1), m_word2(word2)
        {}

        inline bool soi() const { return ((m_word1 >> 13) & 0x1); }
        inline bool ok() const { return ((m_word1 >> 12) & 0x1); }
        inline int adc() const { return (m_word1 & 0xFF); }
        inline int capid() const { return ((m_word2 >> 12) & 0x3); }
        inline int tdc_te() const { return ((m_word2 >> 6) & 0x1F); }
        inline int tdc_le() const { return (m_word2 & 0x3F); }

    private:
        uint16_t m_word1, m_word2;
    };

    inline int nsamples() const { return samples; }
    inline int channelid() const { return ((*data) & 0xFF); }
    inline sample get_sample(int i) {
        uint16_t const *ptr = data + HEADER_WORDS + i*WORDS_PER_SAMPLE;
        return sample(*ptr, *(ptr+1));
    }

private:
    uint16_t const *data;
    int samples;
};

}

#endif
