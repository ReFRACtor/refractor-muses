#!/bin/bash

# Where this script is running from
script_dir=$(readlink -f $(dirname $0))

# Figure out base path to programs so we don't need to chdir to where they live
amuseme_path=$(which amuse-me)

if [ -z "$amuseme_path" ]; then
    echo "Unable to find amuse-me from environment PATH"
    exit 1
fi

if [ -z "$(echo $amuseme_path | grep venv)" ]; then
    echo "amuse-me does not appear to be located within a Python virtual environment, unsure how to deduce MUSES programs path"
    exit 1
fi

programs_root=$(echo $amuseme_path | sed 's/amuse-me.*$//')

if [ ! -d "$programs_root/amuse-me" ]; then
    echo "MUSES programs root path found does not appear to be valid at: $programs_root"
    exit 1
fi

# Make sure OSP path variable is set
if [ -z "$MUSES_OSP_PATH" ]; then
    echo "Please set the \$MUSES_OSP_PATH environment variable to point to the OSP root directory"
fi

# pipeline config has been modified such that setup_targets.yml lists the index
# of the target we wish to reproduce:
# 20160414_23_394_23

echo $amuseme_path \
  --output $script_dir/output \
  --OSP $MUSES_OSP_PATH \
  --programs $programs_root \
  --sensor-set OMI \
  --profile Global_Survey \
  --date 2016-04-14 \
  --python \
  $*

$amuseme_path \
  --output $script_dir/output \
  --OSP $MUSES_OSP_PATH \
  --programs $programs_root \
  --sensor-set OMI \
  --profile Global_Survey \
  --date 2016-04-14 \
  --python \
  $*
