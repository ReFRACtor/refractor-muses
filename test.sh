#! /bin/bash

test_dir=$(dirname $0)/test

PYTHONPATH=$(pwd)/python:${PYTHONPATH} python3 -m pytest -rxXs --log-cli-level=info ${test_dir} $*
