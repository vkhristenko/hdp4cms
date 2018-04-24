#include <iostream>
#include <fstream>
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

#include "dataformats/raw_fed/interface/edm__Wrapper_FEDRawDataCollection_.h"
#include "dataformats/raw_hcal/interface/digi_collection.hpp"
#include "raw2digi/hcal/interface/unpack.hpp"
#include "raw2digi/ecal/interface/unpack.hpp"
#include "raw2digi/pixel/interface/unpack.hpp"
#include "raw2digi/common/interface/fed_numbering.hpp"
#include "raw2digi/common/interface/dump_raw.hpp"

#ifdef USE_OPENCL
#define CL_HPP_ENABLE_EXCEPTIONS
#define CL_HPP_TARGET_OPENCL_VERSION 120
#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#include "execs/raw2digi/interface/cl2.hpp"
#endif // USE_OPENCL

namespace common {

#ifdef USE_OPENCL
using kernel_maker_t = cl::compatibility::make_kernel<cl::Buffer, cl::Buffer, 
                                                    int const, int const,
                                                    float const, int const>;
#endif

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

#ifdef USE_OPENCL
struct clctx_t {
    cl::CommandQueue            queue;
    cl::Context                 ctx;
};
#endif

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

#ifdef USE_OPENCL
void process_hcal_opencl(dataformats::raw_hcal::collections const& hcal_digis,
                         hcal::clctx_t clctx,
                         common::kernel_maker_t kkk) {
    using namespace dataformats::raw_hcal;
    
    auto [digis_f01, digis_f2, digis_f3, digis_f4, digis_f5] = hcal_digis;
    PRINT(digis_f01.size());
    PRINT(digis_f2.size());
    PRINT(digis_f3.size());
    PRINT(digis_f4.size());
    PRINT(digis_f5.size());

    if (digis_f01.size() == 0)
        return;

    // allocate the sums vector right awway
    std::vector<float> test_out_f01(digis_f01.size());

    // setup input/output buffers
    cl::Buffer d_in_f01(clctx.ctx, 
        begin(digis_f01.data()), end(digis_f01.data()), true);
    /*
    cl::Buffer d_in_f2(clctx.ctx, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, 
        sizeof(uint16_t)*digis_f2.data().size(), digis_f2.data().data());
    cl::Buffer d_in_f3(clctx.ctx, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, 
        sizeof(uint16_t)*digis_f3.data().size(), digis_f3.data().data());
    cl::Buffer d_in_f4(clctx.ctx, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, 
        sizeof(uint16_t)*digis_f4.data().size(), digis_f4.data().data());
    cl::Buffer d_in_f5(clctx.ctx, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, 
        sizeof(uint16_t)*digis_f5.data().size(), digis_f5.data().data());
        d_in_f2, d_in_f3, d_in_f4, d_in_f5;
        */
    cl::Buffer d_out_f01(clctx.ctx, CL_MEM_WRITE_ONLY, sizeof(float)*digis_f01.size());
    /*
    cl::Buffer d_out_f2(clctx.ctx, CL_MEM_WRITE_ONLY, sizeof(float)*digis_f2.size());
    cl::Buffer d_out_f3(clctx.ctx, CL_MEM_WRITE_ONLY, sizeof(float)*digis_f3.size());
    cl::Buffer d_out_f4(clctx.ctx, CL_MEM_WRITE_ONLY, sizeof(float)*digis_f4.size());
    cl::Buffer d_out_f5(clctx.ctx, CL_MEM_WRITE_ONLY, sizeof(float)*digis_f5.size());
    */

    // launch kernel
    // test only for the f01
    kkk(cl::EnqueueArgs(clctx.queue, cl::NDRange(digis_f01.size())),
        d_in_f01,
        d_out_f01,
        digis_f01.get_nsamples(),
        data_f01::HEADER_WORDS,
        data_f01::WORDS_PER_SAMPLE,
        digis_f01.size());
    clctx.queue.finish();
    cl::copy(clctx.queue, d_out_f01, begin(test_out_f01), end(test_out_f01));

    // run the checks
    printf("testing opencl sum computation\n");
    PRINT(test_out_f01.size());
    for (auto& sum : test_out_f01) 
        printf("sum = %f\n", sum);
}
#endif // USE_OPENCL

void process_hcal(dataformats::raw_hcal::collections const& hcal_digis) {
    using namespace dataformats::raw_hcal;
    auto [digis_f01, digis_f2, digis_f3, digis_f4, digis_f5] = hcal_digis;

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

#ifdef USE_OPENCL

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
    cl::Program program;
    try {
        program = cl::Program(clctx.ctx, common::loadProgram(progName), false);
        program.build("-cl-std=CL2.0");
    }
    catch (cl::Error& e) {
        printf("err msg: %s\n", e.what());
        if (e.err() == CL_BUILD_PROGRAM_FAILURE) {
            printf("%d error CL build program failure\n", e.err());

            auto buildInfo = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>();
            for (auto& p : buildInfo)
                std::cout << p.second << std::endl;
        }
        exit(1);
    }

    clctx.queue = cl::CommandQueue((clctx.ctx));
    auto mk_vsum_simple = common::kernel_maker_t(
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

#endif // USE_OPENCL
}
