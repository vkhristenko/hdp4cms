#ifdef __CINT__

#pragma link off all globals;
#pragma link off all classes;
#pragma link off all functions;

#pragma link C++ class std::vector<unsigned char>+;
#pragma link C++ class std::vector<std::vector<unsigned char> >+;

#pragma link C++ class edm::DoNotRecordParents+;
#pragma link C++ class FEDRawData+;
#pragma link C++ class FEDRawDataCollection+;
#pragma link C++ class edm::ViewTypeChecker+;
#pragma link C++ class edm::WrapperBase+;
#pragma link C++ class edm::Wrapper<FEDRawDataCollection>+;

#endif
