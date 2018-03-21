#include "/Users/vk/software/heter4cms/hdp4cms/execs/test_opencl/src/def.h"

struct some_data_t {
    float d;
};

__kernel void vadd(                             
    __global float* a,                      
    __global float* b,                      
    __global float* c,                      
    const unsigned int count)               
{                                          
    int i = get_global_id(0);               
//    some_data_t * pdata = (some_data_t*)a;
    for (int j=0; j<1000*100; j++) {
        if(i < count)
            c[i] = a[i] + b[i];                 
//            c[i] = pdata[i].d + b[i];                 
    }
}  
