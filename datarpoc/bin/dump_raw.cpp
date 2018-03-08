#include <iostream>
#include <string>

#define USEROOT

#ifdef USEROOT
#include "TFile.h"
#include "TTree.h"
#endif

int main(int argc, char ** argv) {
    std::cout << "hello world" << std::endl;

    // parse the args
    std::string pathToFile(argv[1]);

    // validate args
    std::cout << "pathToFile = " << pathToFile << std::endl;
}
