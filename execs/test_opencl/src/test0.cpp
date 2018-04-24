#include <cstdio>

#ifdef USE_OPENCL
#include <OpenCL/cl.h>
#endif

int main(int argc, char ** argv) {
    printf("hello world\n");

#ifdef USE_OPENCL
    cl_platform_id clid;
    int result = clGetPlatformIDs(1, &clid, NULL);

    if (result == CL_SUCCESS) {
        printf("true\n");
    }
    else {
        printf("false\n");
    }
#endif // USE_OPENCL

    return 0;
}
