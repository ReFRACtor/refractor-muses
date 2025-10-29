#! /bin/bash

test_dir=$(dirname $0)/tests

PYTHONPATH=$(pwd)/python:${PYTHONPATH} python3 -m pytest -rfxXs --log-cli-level=info ${test_dir} $*
