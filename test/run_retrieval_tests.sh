#!/bin/bash

script_dir=$(readlink -f $(dirname $0))

test_inp_dir=$script_dir/unit_test_data/in
config_dir=$script_dir/../config

config_file=$config_dir/retrieval_run_config.py

export OMI_RUG_FILENAME=$test_inp_dir/OMI-Aura_L1-OML1BRUG_2016m0414t0020-o62484_v003-2016m0414t061442.he4
export OMI_IRR_FILENAME=$test_inp_dir/OMI-Aura_L1-OML1BIRR_2016m0414t0337-o62486_v003-2016m0621t174011.he4

export OMI_ALONG_TRACK_INDEX="704"
export OMI_ACROSS_TRACK_INDEXES="11,23"

sounding_id=${OMI_ALONG_TRACK_INDEX}_$(echo $OMI_ACROSS_TRACK_INDEXES | sed 's/ /_/g' | sed 's/,/_/g' | sed -r 's/_+/_/g')
run_rf_config.py $config_file -o omi_retrieval_${sounding_id}.nc -v 2>&1 | tee omi_retrieval_${sounding_id}.log
