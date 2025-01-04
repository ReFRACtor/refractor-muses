This directory contains all the capture tests. These aren't really
tests, but it is convenient to present them that way so we can run
this with py-test.

These generate various "captured" data, which generally goes to places
in refractor-test-data. This are states after portions of the full
retrieval are done (e.g., after step 1, step 2, etc. of a joint
CrIS/TROPOMI retrieval).

This captured data is then used in other unit tests.

Note it is perfectly fine to run the capture steps in parallel (i.e.,
pytest with -n 10 or whatever). This generates them faster, and
everything is done in its own directory so this is clean.

The capture tests are normally skipped, you need to pass --run-capture on
the command line to have the tests actually run.

These tests assumes the data files are already in refractor_test_data. If not,
you can either manually copy them there, or uses
MusesRunDir.save_run_directory

