#!/bin/bash

script_dir=$(readlink -f $(dirname $0))

bash $script_dir/base_command.sh \
  --start-step geolocate \
  --end-step setup_targets \
