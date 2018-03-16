#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cstdlib>

#define CL_HPP_ENABLE_EXCEPTIONS
#define CL_HPP_TARGET_OPENCL_VERSION 120
#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#include "cl2.hpp"

#define TOL 0.001
#define SIZE 1024 * 1024

#define PRINT_FLOAT(stmt) \
    printf(#stmt " = %f\n", stmt);

// read in the source code
inline std::string loadProgram(std::string input)
{
        std::ifstream stream(input.c_str());
            if (!stream.is_open()) {
                        std::cout << "Cannot open file: " << input << std::endl;
                                exit(1);
                                    }

                 return std::string(
                                 std::istreambuf_iterator<char>(stream),
                                         (std::istreambuf_iterator<char>()));
}

int main(int argc, char** argv) {
    int dtype = 0;
    if (argc!=3) {
        printf("usage: ./vadd /path/to/vadd.cl device_type (0: gpu, 1: cpu)\n");
        return EXIT_FAILURE;
    }
    dtype = atoi(argv[2]);

    std::string progName(argv[1]);

    std::vector<float> h_a(SIZE);
    std::vector<float> h_b(SIZE);
    std::vector<float> h_c(SIZE);

    cl::Buffer d_a;
    cl::Buffer d_b;
    cl::Buffer d_c;

    int const count = SIZE;
    for (auto i =0; i<count; i++) {
        h_a[i] = i;
        h_b[i] = i*i;
    }
    
    // need to have a gpu on the machine
    cl::Context ctx;
    if (dtype==0) {
        ctx = cl::Context(CL_DEVICE_TYPE_GPU);
        printf("using gpu!\n");
    }
    else if (dtype == 1) {
        ctx = cl::Context(CL_DEVICE_TYPE_CPU);
        printf("using cpu\n");
    }
    else {
        ctx = cl::Context(CL_DEVICE_TYPE_GPU);
        printf("using gpu\n");
    }

    // load kernel and create a prgoram object for the cxt provided
    cl::Program program(ctx, loadProgram(progName), true);

    // get a queue
    cl::CommandQueue queue((ctx));

    // create a kernel functor
    auto vadd = cl::compatibility::make_kernel<cl::Buffer, cl::Buffer, cl::Buffer, int>(program, "vadd");

    d_a = cl::Buffer(ctx, begin(h_a), end(h_a), true);
    d_b = cl::Buffer(ctx, begin(h_b), end(h_b), true);
    d_c = cl::Buffer(ctx, CL_MEM_WRITE_ONLY, sizeof(float) * SIZE);

    vadd(cl::EnqueueArgs(queue, cl::NDRange(count)),
         d_a, 
         d_b,
         d_c,
         count);

    queue.finish();

    cl::copy(queue, d_c, begin(h_c), end(h_c));

#if 0
    // testing the output
    int correct = 0;
    int incorrect = 0;
    float tmp;
    for (auto i=0; i<count; i++) {
        tmp = h_a[i] + h_b[i];
        PRINT_FLOAT(h_a[i] + h_b[i]);
        tmp -= h_c[i];
        if (tmp == 0)
            correct++;
        else {
            incorrect++;
            printf("comparison failed\n"); 
        }
    }

    printf("correct = %d, incorrect = %d, total size = %d\n", correct, incorrect,
        SIZE);
#endif
}
