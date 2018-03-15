#include <iostream>
#include <string>
#include <vector>
#include <stdexcept>

#define USEROOT
#define THRESHOLD 2
//##undef USEROOT

#ifdef USEROOT
#include "TFile.h"
#include "TTree.h"
#endif

#include "dataformats/raw_fed/interface/edm__Wrapper_FEDRawDataCollection_.h"
#include "raw2digi/hcal/interface/unpack.hpp"
#include "raw2digi/ecal/interface/unpack.hpp"
#include "raw2digi/pixel/interface/unpack.hpp"
#include "raw2digi/common/interface/fed_numbering.hpp"
#include "raw2digi/common/interface/dump_raw.hpp"

int main(int argc, char ** argv) {
    std::cout << "hello world" << std::endl;

    using namespace raw2digi::common;

    // parse the args
    std::string pathToFile(argv[1]);

    // validate args
    PRINT(pathToFile);

    TFile *f = new TFile(pathToFile.c_str());
    TTree *tree = (TTree*)f->Get("Events");

    // for debug
    PRINT(tree->GetEntries());

    edm::Wrapper<FEDRawDataCollection> *raw = nullptr;
    tree->SetBranchAddress("FEDRawDataCollection_rawDataCollector__LHC.", &raw);

    auto nevents = tree->GetEntries();
    for (auto i=0; i<nevents && i<THRESHOLD; i++) {
        printf("\n\n********************************\n");
        printf("   New Event   \n");
        printf("********************************\n\n");

        // event content
        tree->GetEntry(i);
        PRINT(raw->present);
        std::vector<FEDRawData>& rawcollection = raw->obj.data_;

        PRINT(rawcollection.size());

        for (auto fed=0; fed<rawcollection.size(); fed++) {
            FEDRawData& buffer = rawcollection[fed];
            PRINT(fed);
            PRINT(buffer.data_.size());
            
            if (buffer.data_.size() == 0) {
                printf("skipping fed = %d\n", fed);
                continue;
            }
            if (raw2digi::common::is_hcal_fed(fed))
                raw2digi::hcal::unpack(buffer.data_, fed);
            else if (raw2digi::common::is_ecal_fed(fed))
                raw2digi::ecal::unpack(buffer.data_, fed);
            else if (raw2digi::common::is_pixel_fed(fed))
                raw2digi::pixel::unpack(buffer.data_, fed);
            else
                printf("UNKNOWN FED fed=%d", fed);
        }

        printf("\n\n********************************\n");
        printf("   End of Event   \n");
        printf("********************************\n\n");
    }

    f->Close();
}
