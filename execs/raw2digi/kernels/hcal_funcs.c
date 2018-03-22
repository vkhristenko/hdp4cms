//#include <inttypes.h>
//#include <stdint.h>

//
// funcs for data_f01
//
int f_data_f01_adc(unsigned short);
int f_data_f01_adc(unsigned short data) {
    return data & 0x00FF;
}

// 
// simple sum kernel
//
__kernel void vsum_simple(__global unsigned short* digis,
                          __global float* sums,
                          int const nsamples,
                          int const nheader_words,
                          float const nwords_per_sample,
                          int const size) {
    // digi id
    int idigi =get_global_id(0);
    // start of the raw data position
    int ipos = idigi * (nheader_words + nsamples*nwords_per_sample);

    // set the pointer to the start of the digi
    __global unsigned short* data = digis + ipos + nheader_words;
    float sum = 0;
    for (int is=0; is<nsamples; is++) {
        sum += f_data_f01_adc(*(data + int(is*nwords_per_sample)));
    }

    sums[idigi] = sum;
}
