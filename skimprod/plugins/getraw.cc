// -*- C++ -*-
//
// Package:    analytics4cmssw/getraw
// Class:      getraw
// 
/**\class getraw getraw.cc analytics4cmssw/getraw/plugins/getraw.cc

 Description: [one line class summary]

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Viktor Khristenko
//         Created:  Thu, 08 Mar 2018 16:17:34 GMT
//
//


// system include files
#include <memory>

// ROOT includes
#include "TTree.h"

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/FEDRawData/interface/FEDNumbering.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"

#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

//
// class declaration
//

// If the analyzer does not use TFileService, please remove
// the template argument to the base class so the class inherits
// from  edm::one::EDAnalyzer<> and also remove the line from
// constructor "usesResource("TFileService");"
// This will improve performance in multithreaded jobs.

class getraw : public edm::one::EDAnalyzer<>  {
   public:
      using TRawDataCollection = std::vector<std::vector<unsigned char>>;
      explicit getraw(const edm::ParameterSet&);
      ~getraw();

      static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);


   private:
      virtual void beginJob() override;
      virtual void analyze(const edm::Event&, const edm::EventSetup&) override;
      virtual void endJob() override;

      // ----------member data ---------------------------
      edm::EDGetTokenT<FEDRawDataCollection> m_tRawCollection;

      TTree                                     *m_tree;
      TTree                                     *m_treeAux;
      TRawDataCollection                        m_raw;
      std::vector<int>                          m_feds;
};

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
getraw::getraw(const edm::ParameterSet& iConfig)
{
    // token for the raw collection
    m_tRawCollection = consumes<FEDRawDataCollection>(
        iConfig.getParameter<edm::InputTag>("InputLabel"));

    // retrieve the list of feds to unpack for hcal
    for (int i=FEDNumbering::MINHCALuTCAFEDID; i<=FEDNumbering::MAXHCALuTCAFEDID; i++)
        m_feds.push_back(i);
    for (int i=FEDNumbering::MINECALFEDID; i <= MINCASTORFEDID; i++)
        m_feds.push_back(i);

    edm::Service<TFileService> fs;
    m_tree =fs->make<TTree>("Events", "Events");
    m_tree->Branch("RawData", (TRawDataCollection*)&m_raw);

    // ot be filled just once
    m_treeAux = fs->make<TTree>("Aux", "Aux");
    m_treeAux->Branch("FEDs", (std::vector<int>*)&m_feds);
    m_treeAux->Fill();
}


getraw::~getraw()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called for each event  ------------
void
getraw::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
    // extract the raw collection
    edm::Handle<FEDRawDataCollection> hRawCollection;
    iEvent.getByToken(m_tRawCollection, hRawCollection);

    for (std::vector<int>::const_iterator it=m_feds.begin(); 
        it < m_feds.end(); ++it) {
        // 
        // retrieve the fed raw buffer
        //
        const FEDRawData& fdata = hRawCollection->FEDData(*it);
        unsigned char *data = (unsigned char*) fdata.data();
        m_raw.emplace_back(data, data + fdata.size());
    }

    m_tree->Fill();
    m_raw.clear();
}


// ------------ method called once each job just before starting event loop  ------------
void 
getraw::beginJob()
{
}

// ------------ method called once each job just after ending the event loop  ------------
void 
getraw::endJob() 
{
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void
getraw::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(getraw);
