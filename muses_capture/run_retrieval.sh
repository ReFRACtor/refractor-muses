script_dir=$(readlink -f $(dirname $0))

sh $script_dir/base_command.sh \
  --start-step retrieve_one \
  --end-step retrieve_one \
  --retrieve-one 20160414_23_394_11_23
