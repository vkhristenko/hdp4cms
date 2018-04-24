#include <iostream>
#include <vector>
#include <Eigen/Dense>

using namespace Eigen;
using namespace std;

constexpr int MaxSVSize=10;
constexpr int MaxFSVSize=19;
constexpr int MaxPVSize=10;

typedef Eigen::Matrix<double,Eigen::Dynamic,1,0,MaxSVSize,1> SampleVector;
typedef Eigen::Matrix<double,Eigen::Dynamic,1,0,MaxPVSize,1> PulseVector;
typedef Eigen::Matrix<int,Eigen::Dynamic,1,0,MaxPVSize,1> BXVector;

typedef Eigen::Matrix<double,MaxFSVSize,1> FullSampleVector;
typedef Eigen::Matrix<double,MaxFSVSize,MaxFSVSize> FullSampleMatrix;

typedef Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic,0,MaxSVSize,MaxSVSize> SampleMatrix;
typedef Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic,0,MaxPVSize,MaxPVSize> PulseMatrix;
typedef Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic,0,MaxSVSize,MaxPVSize> SamplePulseMatrix;

typedef Eigen::LLT<SampleMatrix> SampleDecompLLT;
typedef Eigen::LLT<PulseMatrix> PulseDecompLLT;
typedef Eigen::LDLT<PulseMatrix> PulseDecompLDLT;

typedef Eigen::Matrix<double,1,1> SingleMatrix;
typedef Eigen::Matrix<double,1,1> SingleVector;

int main() {
    std::cout << "hello world" << std::endl;

    MatrixXd m = MatrixXd::Random(3,3);
    m = (m + MatrixXd::Constant(3,3,1.2))*50;
    cout << "m = " << endl << m << endl;

    VectorXd v(3);
    v << 1,2,3;
    cout << "m*v = " << endl << m*v << endl;

    vector<float> vd {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    Map<Matrix<float, Dynamic, Dynamic, RowMajor>> M(vd.data(), 3,3);
    cout << "M = " << endl << M << endl;
}
