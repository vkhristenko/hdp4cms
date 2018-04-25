#include <iostream>

#ifdef USE_CUDA
#include "cuda.h"
#include "cuda_runtime.h"
#endif

int main() {
    std::cout << "hello world" << std::endl;

#ifdef USE_CUDA
    int nDevices;
    cudaGetDeviceCount(&nDevices);
    std::cout << "nDevices = " << nDevices << std::endl;
#endif // USE_CUDA
}
