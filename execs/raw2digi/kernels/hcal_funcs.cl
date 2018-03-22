

// 
// simple sum
//
__kernel void vsum_simple(__global uint16_t* digis,
                          __global float* sums,
                          int nsamples,
                          int size) {
    int i = get_global_id(0);
}
