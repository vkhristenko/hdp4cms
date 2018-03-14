#include <iostream>

#include "raw2digi/hcal/interface/unpack_utca.hpp"
#include "raw2digi/common/interface/dump_raw.hpp"
#include "dataformats/raw_hcal/interface/amc13header.hpp"
#include "dataformats/raw_hcal/interface/uhtr_data.hpp"
#include "dataformats/raw_hcal/interface/channel_data.hpp"

namespace raw2digi::hcal {

using namespace dataformats::raw_hcal;
using namespace dataformats::raw_fed;
using namespace raw2digi::common;

void unpack_utca(dataformats::raw_fed::raw_buffer const& buffer) {
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
            else if (it.flavor() == 3) {
                data_f3 ch_data(it(), (it+1)());

                PRINT(ch_data.channelid());
                PRINT(ch_data.nsamples());

                printf("Sample: \n");
                for (auto is=0; is<ch_data.nsamples(); is++) {
                    PRINT(ch_data.get_sample(is).soi());
                    PRINT(ch_data.get_sample(is).le());
                    PRINT(ch_data.get_sample(is).capid());
                    PRINT(ch_data.get_sample(is).adc());
                    PRINT(ch_data.get_sample(is).tdc());
                }
            }
            else if (it.flavor() == 4) {
                data_f4 ch_data(it(), (it+1)());

                PRINT(ch_data.channelid());
                PRINT(ch_data.nsamples());

                printf("Sample: \n");
                for (auto is=0; is<ch_data.nsamples(); is++) {
                    PRINT(ch_data.get_sample(is).soi());
                    PRINT(ch_data.get_sample(is).ok());
                    PRINT(ch_data.get_sample(is).tpg());
                }
            }
            else if (it.flavor() == 5) {
                data_f5 ch_data(it(), (it+1)());

                PRINT(ch_data.channelid());
                PRINT(ch_data.nsamples());

                printf("Sample: \n");
                for (auto is=0; is<ch_data.nsamples(); is++) {
                    PRINT(ch_data.get_sample(is).adc());
                }
            }
            else
                continue;
        }

        // cast the payload to unsigned char *
        unsigned char const * payload_tmp = reinterpret_cast<unsigned char const*>(payload);
        raw_buffer buffer_tmp(payload_tmp, payload_tmp + size*8);
        // dump the paylaad
        printf("\n\n***********************************\n");
        printf("    Dumping RAW Payload __only__ Buffer    size = %dB\n",
               size * 8);
        printf("***********************************\n\n");
        // dump 16 bit words per line
        // as in hcal specification!
        common::dump_raw(buffer_tmp, 2);
    }
}

}
