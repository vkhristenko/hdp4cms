#include <iostream>
#include <string>
#include <vector>

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

// type decls
using TRawBuffer = std::vector<unsigned char>;
using TRawDataCollection = std::vector<TRawBuffer>;

namespace common {

void dump_raw(TRawBuffer const& buffer) {
    auto counter = 0;
    for (auto it = buffer.begin(); it!=buffer.end(); counter++, ++it) {
        if (counter % 25 == 0) 
            printf("\n");
        printf("%02x ", *it);
    }
    printf("\n\n");
}

}

namespace hcal {

class uhtr_data {
public:

private:
    int formatversion;
    int rawlength64;
    uint64_t const* raw64;
    uint16_t const* raw16;
    uint64_t* data;
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
        
        // cast the payload to unsigned char *
        unsigned char const * payload_tmp = reinterpret_cast<unsigned char const*>(payload);
        TRawBuffer buffer_tmp(payload_tmp, payload_tmp + size*8);

        // dump the paylaad
        printf("\n\n***********************************\n");
        printf("    Dumping RAW Payload __only__ Buffer    size = %d\n", 
               size * 8);
        printf("***********************************\n\n");
        common::dump_raw(buffer_tmp);
    }

    // num of uhtrs in this FED
    /*
    auto nuhtrs = header->namc();
    for (auto iuhtr=0; i<nuhtrs; i++) {

    }*/
}

void unpack(TRawBuffer const& buffer) {
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
    printf("    Dumping the whole RAW Buffer    size = %lu\n", buffer.size());
    printf("***********************************\n\n");
    common::dump_raw(buffer);
}

}// end namespace hcal

int main(int argc, char ** argv) {
    std::cout << "hello world" << std::endl;

    // parse the args
    std::string pathToFile(argv[1]);

    // validate args
    PRINT(pathToFile);

    TFile *f = new TFile(pathToFile.c_str());
    TTree *tree = (TTree*)f->Get("getraw/Events");

    // for debug
    PRINT(tree->GetEntries());

    TRawDataCollection *raw = nullptr;

    tree->SetBranchAddress("RawData", &raw);

    int nevents = tree->GetEntries();
    for (auto i=0; i<nevents && i<THRESHOLD; i++) {
        tree->GetEntry(i);
        PRINT(raw->size());

        for (auto it=raw->begin(); it!=raw->end(); ++it) {
            if (it->size() == 0) {
                printf("skipping fed\n");
                continue;
            }
            hcal::unpack(*it);
        }
    }

    f->Close();
}
