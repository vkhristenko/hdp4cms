//
// simple vector addition example
//
__global__ void vector_add(int* a, int* b, int* c) {
    int id = blockIdx.x;
    c[id] = a[id] + b[id];
}
