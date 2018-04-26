#include "execs/test_cuda_eigen/interface/test_eigen_kernels0.h"


__global__ void cu_eigen_vadd(Eigen::Vector3d* va,
                         Eigen::Vector3d* vb,
                         Eigen::Vector3d* vc) {
    int id = blockIdx.x;
    vc[id] = va[id] + vb[id];
}

void eigen_vector_add(Eigen::Vector3d* va, 
                      Eigen::Vector3d* vb, 
                      Eigen::Vector3d* vc, int const n) {
    cu_eigen_vadd<<<n, 1>>>(va, vb, vc);
}
