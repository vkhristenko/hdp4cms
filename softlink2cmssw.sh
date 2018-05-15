#!/bin/bash

CMSSW_BASE=$1
BASE=`dirname ${BASH_SOURCE[0]}`

#
# create soft links for all the directories 
#
for f in dataformats raw2digi; do
    ln -s $BASE/$f $CMSSW_BASE/$f
done
