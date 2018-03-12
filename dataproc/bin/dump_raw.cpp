#include <iostream>
#include <string>
#include <vector>
#include <stdexcept>

#define USEROOT
#define THRESHOLD 10
//##undef USEROOT

#ifdef USEROOT
#include "TFile.h"
#include "TTree.h"
#endif

#define PRINT(x) \
    std::cout << #x " = " << x  << std::endl

#define PRINTRAW(x) \
    printf(#x " = %02x\n", x);

#define MINECALFEDID 600
#define MAXECALFEDID 670
#define MINHCALuTCAFEDID 1100
#define MAXHCALuTCAFEDID 1199

// type decls
using TRawBuffer = std::vector<unsigned char>;
using TRawDataCollection = std::vector<TRawBuffer>;

namespace common {

bool is_hcal_fed(int fed) {
    return fed>=MINHCALuTCAFEDID && fed<=MAXHCALuTCAFEDID;
}

bool is_ecal_fed(int fed) {
    return fed>=MINECALFEDID && fed<=MAXECALFEDID;
}

void dump_raw(TRawBuffer const& buffer, int countermax = 25) {
    auto counter = 0;
    for (auto it = buffer.begin(); it!=buffer.end(); counter++, ++it) {
        if (counter % countermax == 0) 
            printf("\n");
        printf("%02x ", *it);
    }
    printf("\n\n");
}

}

namespace hcal {


// data for flavors 0 or 1
// class is located outside of the uhtr_data as the same payload (byte layout)
// might sit in the VME payload - TO BE SEEN...
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
    uint64_t cdf_header;
    uint64_t amc13_header;
    module_header modules_headers[12];
};

void unpack_utca(TRawBuffer const& buffer) {
    // raw buffer 
    unsigned char const *data = &buffer[0];

    // emulate the cmssw process
    amc13header const *header = (amc13header const*)data;

    PRINT(header->namc());
    PRINT(header->amc13formatversion());
    PRINT(header->sourceid());
    PRINT(header->bunchid());
    PRINT(header->l1anumber());
    PRINT(header->cdfeventtype());
    PRINT(header->orbitnumber());

    for (auto iamc=0; iamc<header->namc(); iamc++) {
        // some additional defs
        int crate = header->getm(iamc).amcid() & 0xFF;
        int nps = (header->getm(iamc).amcid() >> 12) & 0xF;

        PRINT(iamc);
        PRINT(crate);
        PRINT(nps);
        PRINT(header->getm(iamc).amcid());
        PRINT(header->getm(iamc).amcslot());
        PRINT(header->getm(iamc).amcblocknumber());
        PRINT(header->getm(iamc).amcsize());
        PRINT(header->getm(iamc).amcmore());
        PRINT(header->getm(iamc).amcsegmented());
        PRINT(header->getm(iamc).amclengthok());
        PRINT(header->getm(iamc).amccrok());
        PRINT(header->getm(iamc).amcdatapresent());
        PRINT(header->getm(iamc).amcenabled());

        // get the payload and size
        uint64_t const *payload = header->payload(iamc);
        auto size = header->getm(iamc).amcsize();
        uhtr_data uhtr(payload, size);
        PRINT(uhtr.get_headerv1()->bcn());
        PRINT(uhtr.get_headerv1()->evn());
        PRINT(uhtr.get_headerv1()->presamples());
        PRINT(uhtr.get_headerv1()->slotid());
        PRINT(uhtr.get_headerv1()->crateid());
        PRINT(uhtr.get_headerv1()->orb());
        PRINT(uhtr.get_format_version());

        // TODO: eliminate this issue with uMNio 
        if (uhtr.get_format_version() != 1)
            continue;

        for (auto it=uhtr.begin(); it!=uhtr.end(); ++it) {
            PRINT(it.flavor());
            if (it.flavor() == 0 || it.flavor() == 1) {
                // flavor 0 or 1
                data_f01 ch_data(it(), (it+1)());

                // debug
                PRINT(ch_data.channelid());
                PRINT(ch_data.capid());
                PRINT(ch_data.nsamples());
                printf("Sample: \n");
                for (auto is=0; is<ch_data.nsamples(); is++) {
                    PRINT(ch_data.get_sample(is).soi());
                    PRINT(ch_data.get_sample(is).adc());
                    PRINT(ch_data.get_sample(is).tdc());
                }
            } 
            else if (it.flavor() == 2) {
                data_f2 ch_data(it(), (it+1)());

                PRINT(ch_data.channelid());
                PRINT(ch_data.nsamples());

                printf("Sample: \n");
                for (auto is=0; is<ch_data.nsamples(); is++) {
                    PRINT(ch_data.get_sample(is).soi());
                    PRINT(ch_data.get_sample(is).ok());
                    PRINT(ch_data.get_sample(is).adc());
                    PRINT(ch_data.get_sample(is).tdc_te());
                    PRINT(ch_data.get_sample(is).tdc_le());
                }
            }
            else 
                continue;
        }
        
        // cast the payload to unsigned char *
        unsigned char const * payload_tmp = reinterpret_cast<unsigned char const*>(payload);
        TRawBuffer buffer_tmp(payload_tmp, payload_tmp + size*8);
        // dump the paylaad
        printf("\n\n***********************************\n");
        printf("    Dumping RAW Payload __only__ Buffer    size = %dB\n", 
               size * 8);
        printf("***********************************\n\n");
        // dump 16 bit words per line
        // as in hcal specification!
        common::dump_raw(buffer_tmp, 2);
    }

    // num of uhtrs in this FED
    /*
    auto nuhtrs = header->namc();
    for (auto iuhtr=0; i<nuhtrs; i++) {

    }*/
}

void unpack(TRawBuffer const& buffer, int fed) {
    PRINT(buffer.size());

    // pointer to the first byte of the buffer
    unsigned char const *data = &buffer[0];

    // BOEshouldBeZeroAlways() const { return ( (commondataformat3>>28) & 0x0F); }
    // /** Get the Beginning Of Event bits.  If it's not the first or last CDF Slink64 word, the high 4 bits must be zero.*/
    short boe = (data[12] >> 4) & 0x0F;
    PRINTRAW(data[12]);
    PRINTRAW(boe);

    if (boe == 0) {
        printf("vme fed found\n");
        return;
    }
    unpack_utca(buffer);

    // dump the whole buffer
    printf("\n\n***********************************\n");
    printf("    FED=%d Dumping the whole RAW Buffer size = %lu Bytes\n", fed, buffer.size());
    printf("***********************************\n\n");
    common::dump_raw(buffer, 8);
}

}// end namespace hcal

namespace ecal {

void unpack(TRawBuffer const& buffer, int fed) {
    // dump the whole buffer
    printf("\n\n***********************************\n");
    printf("    FED=%d Dumping the whole RAW Buffer size = %lu Bytes\n", fed, buffer.size());
    printf("***********************************\n\n");
    common::dump_raw(buffer, 8);

}

} // end of namepsace ecal

int main(int argc, char ** argv) {
    std::cout << "hello world" << std::endl;

    // parse the args
    std::string pathToFile(argv[1]);

    // validate args
    PRINT(pathToFile);

    TFile *f = new TFile(pathToFile.c_str());
    TTree *tree = (TTree*)f->Get("getraw/Events");
    TTree *tree_aux = (TTree*)f->Get("getraw/Aux");

    // for debug
    PRINT(tree->GetEntries());

    TRawDataCollection *raw = nullptr;
    std::vector<int> *feds = nullptr;

    tree->SetBranchAddress("RawData", &raw);
    tree_aux->SetBranchAddress("FEDs", &feds);
    tree_aux->GetEntry(0);

    int nevents = tree->GetEntries();
    for (auto i=0; i<nevents && i<THRESHOLD; i++) {
        printf("\n\n********************************\n");
        printf("   New Event   \n");
        printf("********************************\n\n");

        tree->GetEntry(i);
        PRINT(raw->size());

        for (auto ifed=0; ifed < feds->size(); ifed++) {
            auto fed = feds->at(ifed);
            TRawBuffer const& buffer = raw->at(ifed);
            if (buffer.size() == 0) {
                printf("skipping fed = %d fed %d\n", ifed, fed);
                continue;
            }
            if (common::is_hcal_fed(fed))
                hcal::unpack(buffer, fed);
            else if (common::is_ecal_fed(fed))
                ecal::unpack(buffer, fed);
            else
                printf("UNKNOWN FED fed=%d", fed);
        }
        
        printf("\n\n********************************\n");
        printf("   End of Event   \n");
        printf("********************************\n\n");
    }

    f->Close();
}
