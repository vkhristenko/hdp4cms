# RAW Format Specification

## HCAL DAQ Readout
- 64 bits for FED Header
- 64 bits for AMC13 Header
- 64 bits x 12 for module headers (UHTRs)
  - only 6 typically per AMC13, but can be determined from the data itself
- [CMS HCAL Documentation](http://cmsdoc.cern.ch/cms/HCAL/document/)
  - [CMS HCAL UHTR Readout Specs](https://cms-docdb.cern.ch/cgi-bin/DocDB/RetrieveFile?docid=12306&filename=uhtr_spec.pdf&version=19)
