#ifndef dataformats_raw_hcal_uhtr_data_h
#define dataformats_raw_hcal_uhtr_data_h

#include <cstdio>
#include <stdexcept>

namespace dataformats::raw_hcal {

class uhtr_data {
public:
    uhtr_data(uint64_t const* p64, int n):
        payload64(p64), payload16((uint16_t const*)p64), size64(n)
    {}

    ~uhtr_data()
    {}

    // class headerv1
    class headerv1 {
    public:
        inline int bcn() const { return int((words[1] >> 4) & 0xFFF); }
        inline int evn() const { return int(((uint32_t const*)(words))[1] & 0xFFFFFF); }
        inline int presamples() const { return int((words[4] >> 12) &0xF); }
        inline int slotid() const { return int((words[4] >> 8) & 0xF); }
        inline int crateid() const { return int((words[4] & 0xFF)); }
        inline int orb() const { return int(words[5]); }

    private:
        uint16_t words[8];
    };

    class const_iterator {
    public:
        const_iterator(uint16_t const* ptr, uint16_t const* p_end) :
            word(ptr), end(p_end)
        {}
        ~const_iterator()
        {}

        inline bool is_header() const { return ((*word) & 0x8000)!=0; }
        inline bool is_end() const { return word == end; }
        inline int flavor() const { return (((*word) >> 12) & 0x7); }

        inline uint16_t const* operator()() const { return word; }
        inline bool operator==(const_iterator const& rhs) const { return word == rhs.word; }
        inline bool operator!=(const_iterator const& rhs) const { return word != rhs.word; }
        inline const_iterator operator+(int i) {
            auto it = const_iterator(word, end);
            for (auto ii=0; ii<i; ii++)
                it.operator++();
            return it;
        }
        const_iterator& operator++() {
            // the must
            if (!is_header()) {
                printf("word address = %p\n", word);
                printf("end address = %p\n", end);
                printf("word = %x\n", *word);
                throw std::runtime_error("const_iterator points to a non-header word");
            }
            if (word == end)
                return *this;

            // skip until the next header or until the end of the per channel data
            word++;
            while(!is_header() && !is_end())
                word++;

            return *this;
        }

    private:
        // must always point to the header of the flavored data
        uint16_t const *word;
        // if word == end, done
        uint16_t const *end;
    };

    headerv1 const* get_headerv1() const { return (headerv1 const*)(payload16); }
    const_iterator begin() const {
        return const_iterator(payload16+8,
                              payload16 + (size64-1)*sizeof(uint64_t)/sizeof(int16_t));
    }
    const_iterator end() const {
        return const_iterator(payload16 + (size64-1)*sizeof(uint64_t)/sizeof(int16_t),
                              payload16 + (size64-1)*sizeof(uint64_t)/sizeof(int16_t));
    }

    inline int get_format_version() const { return ((payload16[6] >> 12) & 0xF); }

private:
    uint64_t const *payload64;
    uint16_t const *payload16;
    int size64;
};

}

#endif
