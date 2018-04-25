# hdp4cms
Heterogeneous Data Processing for the CMS Experiment

## Requirements
- cmake
- root

## Installation
- `git clone https://github.com/vkhristenko/hdp4cms`
- `cd hdp4cms; mkdir bin`
- `cmake ../`
- `make`

## Installation on vinavx2
```
ROOT_DIR=/cvmfs/cms.cern.ch/slc7_amd64_gcc700/lcg/root/6.10.09-omkpbe4/cmake cmake -DCMAKE_CXX_COMPILER=g++ -DEIGEN_HOME=/data/vkhriste/eigen-git-mirror  -DCUDA_TOOLKIT_ROOT_DIR=/cvmfs/cms.cern.ch/slc7_amd64_gcc630/external/cuda/9.1.85-cms ../
```

## configurables
- `-DEIGEN_HOME` for the top eigen dir
