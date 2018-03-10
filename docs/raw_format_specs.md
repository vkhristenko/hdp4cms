# RAW Format Specification

## Hcal Readout
- 64 bits for FED Header
- 64 bits for AMC13 Header
- 64 bits x 12 for module headers (UHTRs)
  - only 6 typically per AMC13, but can be determined from the data itself

