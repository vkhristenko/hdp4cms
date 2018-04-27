#include <iostream>
#include <cstdio>
#include <vector>
#include <algorithm>

#if defined __APPLE__
#include <arrayfire.h>
#else 
#include <arrayfire.h>
#endif

std::vector<float> input(100);
double unifRand()
{
    return rand() / double(RAND_MAX);
}

void testBackend()
{
    af::info();
    af::dim4 dims(10, 10, 1, 1);
    af::array A(dims, &input.front());
    af_print(A);
    af::array B = af::constant(0.5, dims, f32);
    af_print(B);
}

void testCholesky() {
    int n = 10;
    af::array t = af::randu(n, n);
    af::array in = af::matmulNT(t, t) + af::identity(n, n) * n;
    af::array out;
    af::cholesky(out, in, true);
    af_print(in);
    af_print(out);
}

int main() {
    std::cout << "hello world" << std::endl;

    std::generate(input.begin(), input.end(), unifRand);
    try {
        printf("trying cpu backend\n");
        af::setBackend(AF_BACKEND_CPU);
        testBackend();

        testCholesky();
    } catch (af::exception& e) {
        printf("caught exception when trying cpu backend");
        fprintf(stderr, "%s\n", e.what());
    }

    try {
        printf("trying cuda backedn\n");
        af::setBackend(AF_BACKEND_CUDA);
        testBackend();
        testCholesky();
    } catch (af::exception& e) {
        printf("caught exception when trying cuda backend\n");
        fprintf(stderr, "%s\n", e.what());
    }

    try {
        printf("trying opencl backend\n");
        af::setBackend(AF_BACKEND_OPENCL);
        testBackend();
        testCholesky();
    } catch (af::exception& e) {
        printf("caught exception when trying opencl backend\n");
        fprintf(stderr, "%s\n", e.what());
    }
}
