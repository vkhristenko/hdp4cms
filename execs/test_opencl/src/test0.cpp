#include <cstdio>

#include <OpenCL/cl.h>

int main(int argc, char ** argv) {
    printf("hello world\n");

    cl_platform_id clid;
    int result = clGetPlatformIDs(1, &clid, NULL);

    if (result == CL_SUCCESS) {
        printf("true\n");
    }
    else {
        printf("false\n");
    }

    return 0;
}
