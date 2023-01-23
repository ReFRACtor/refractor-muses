This directory uses MUSES's amuse-me tool to setup and run the target
used for capturing data used in ReFRACtor unit testing.

1. First set up the MUSES execution environment per the MUSES instructions
   (e.g., use build_supplement).

2. Make sure amuse-me in on your path (e.g.
   export PATH=/home/smyth/muses/amuse-me/bin:$PATH)

3. Make sure libgfortran.so.4 is in the path (needed by vlidort_cli)
   export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
   (this is temporary, see https://jpl.slack.com/archives/CVBUUE5T5/p1664476320620079)

2. Run setup_target.sh to create files needed by run_retrieval.sh.

3. Run run_retrieval.sh

All files will be locate in the output/ sub directory of this directory.

Once these two scripts have finished the retrieval_test.py test_capture_retrieval_data and test_capture_uip tests can be run using the --run-capture argument to py.test:

4. Capture tests. We preface all the capture tests with "test_capture", so
   you can run the full set with
   pytest -rxXs test/ -k test_capture --run-capture

You can see all tests (without running them) by

   pytest --collect-only test/*_test.py -k test_capture --run-capture

Individual tests (as of 10/19/2022, the list might grow).

   pytest -rxXs test/retrieval_test.py -k test_capture_uip[1] --run-capture
   pytest -rxXs test/retrieval_test.py -k test_capture_uip[2] --run-capture
   pytest -rxXs test/retrieval_test.py -k test_capture_retrieval_data[1] --run-capture
   pytest -rxXs test/retrieval_test.py -k test_capture_retrieval_data[2] --run-capture
   pytest -rxXs test/refractor_fm_Test.py -k test_capture_fm_wrapper[1] --run-capture
   pytest -rxXs test/refractor_fm_Test.py -k test_capture_fm_wrapper[2] --run-capture
   
   pytest -rxXs test/radiance_test.py -k test_capture_atmposphere --run-capture
   pytest -rxXs test/radiance_test.py -k test_capture_expected_xsec --run-capture
