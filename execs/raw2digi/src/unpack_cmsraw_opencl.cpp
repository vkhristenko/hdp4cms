#include <iostream>
#include <fstream>
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
#include "dataformats/raw_hcal/interface/digi_collection.hpp"
#include "raw2digi/hcal/interface/unpack.hpp"
#include "raw2digi/ecal/interface/unpack.hpp"
#include "raw2digi/pixel/interface/unpack.hpp"
#include "raw2digi/common/interface/fed_numbering.hpp"
#include "raw2digi/common/interface/dump_raw.hpp"

#define CL_HPP_ENABLE_EXCEPTIONS
#define CL_HPP_TARGET_OPENCL_VERSION 120
#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#include "execs/raw2digi/interface/cl2.hpp"

namespace common {

inline std::string loadProgram(std::string input) {
    std::ifstream stream(input.c_str());
    if(!stream.is_open()){
        std::cout << "Cannot open file: " << input << std::endl;
        exit(1);
    }
    
    return std::string(std::istreambuf_iterator<char>(stream),
                       std::istreambuf_iterator<char>());
}

}

namespace hcal {

struct clctx_t {
    cl::CommandQueue            queue;
    cl::Context                 ctx;
};

template<typename T>
float f_sum(T&);

template<>
float f_sum<dataformats::raw_hcal::data_f4>(dataformats::raw_hcal::data_f4& d) {
    float sum = 0;
    for (auto i=0; i<d.nsamples(); i++)
        sum += d.get_sample(i).tpg();
    return sum;
}

template<typename T>
float f_sum(T& d) {
    float sum = 0;
    for (auto i=0; i<d.nsamples(); i++)
        sum += d.get_sample(i).adc();
    return sum;
}

}

void process_hcal_opencl(dataformats::raw_hcal::collections const& hcal_digis,
                         hcal::clctx_t clctx,
                         cl::compatibility::make_kernel<cl::Buffer, cl::Buffer, int, int> mk_kernel) {
    return;
}

void process_hcal(dataformats::raw_hcal::collections const& hcal_digis) {
    using namespace dataformats::raw_hcal;
    digi_collection_f01 digis_f01;
    digi_collection_f2 digis_f2;
    digi_collection_f3 digis_f3;
    digi_collection_f4 digis_f4;
    digi_collection_f5 digis_f5;

    std::tie(digis_f01, digis_f2, digis_f3, digis_f4, digis_f5) = hcal_digis;
    PRINT(digis_f01.size());
    PRINT(digis_f2.size());
    PRINT(digis_f3.size());
    PRINT(digis_f4.size());
    PRINT(digis_f5.size());

    for (auto i=0; i<digis_f01.size(); i++) {
        auto data = digis_f01[i];
        float sum = hcal::f_sum<data_f01>(data);
        PRINT(sum);
    }
    
    for (auto i=0; i<digis_f2.size(); i++) {
        auto data = digis_f2[i];
        float sum = hcal::f_sum<data_f2>(data);
        PRINT(sum);
    }
    for (auto i=0; i<digis_f3.size(); i++) {
        auto data = digis_f3[i];
        float sum = hcal::f_sum<data_f3>(data);
        PRINT(sum);
    }
    for (auto i=0; i<digis_f4.size(); i++) {
        auto data = digis_f4[i];
        float sum = hcal::f_sum<data_f4>(data);
        PRINT(sum);
    }
    for (auto i=0; i<digis_f5.size(); i++) {
        auto data = digis_f5[i];
        float sum = hcal::f_sum<data_f5>(data);
        PRINT(sum);
    }
}

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

    // set up opencl 
    int error = 0;
    hcal::clctx_t clctx;
    std::string progName(argv[2]);
    clctx.ctx = cl::Context(CL_DEVICE_TYPE_GPU);
    cl::Program program(clctx.ctx, common::loadProgram(progName), true, &error);
    clctx.queue = cl::CommandQueue((clctx.ctx));
    auto mk_vsum_simple = 
        cl::compatibility::make_kernel<cl::Buffer, cl::Buffer, int, int>(
            program, "vsum_simple");

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
            
            // skip a fed with an empty buffer
            if (buffer.data_.size() == 0) {
                printf("skipping fed = %d\n", fed);
                continue;
            }

            // init
            dataformats::raw_hcal::collections hcal_digis;

            // dispatch
            if (raw2digi::common::is_hcal_fed(fed)) {
                hcal_digis = raw2digi::hcal::unpack(buffer.data_, fed);
                process_hcal(hcal_digis);
                process_hcal_opencl(hcal_digis, clctx, mk_vsum_simple);
            }
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
