#include <iostream>

#if defined __APPLE__
#include <Eigen/Dense>
#else 
#include <Eigen/Dense>
#endif

int main() {
    std::cout << "hello world" << std::endl;

    Eigen::MatrixXd m(2,2);
    m(0,0) = 3;
    m(1,0) = 2.5;
    m(0,1) = -1;
    m(1,1) = m(1,0) + m(0,1);
    std::cout << m << std::endl;
}
