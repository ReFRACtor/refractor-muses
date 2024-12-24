Pipeline
========

The MUSE's pipeline is great for running large number of end to end soundings.
But we want to extract out small pieces of this for local system and unit
testing of ReFRACTor.

**Note that this code was used for the initial collection**. Now we have pytests
in test/capture_data_test.py that can be run to duplicate this functionality.
I find it easier to do updates through these capture tests. We'll leave the
original muses-capture program here in case we need to come back to them, but
this isn't usually run now.

This directory uses MUSES's amuse-me tool to setup and run the target
used for capturing data used in ReFRACtor unit testing. This runs the
full pipeline.

The output of the run goes into the directory "output". This will be
local to your directory, it doesn't get checked into git. This directory
is then used by the various capture tests mentioned below (and the
"capture" data does get checked into git.

To run:

1. First set up the MUSES execution environment per the 
   refactor/build_supplement directions (https://github.jpl.nasa.gov/refractor/build_supplement/blob/master/doc/scf_deployment.md)

2. While omi runs use the new muses-vlidort,  tropomi still uses a version
   in the OSP directory. For this one, there is a bug for libgfortran.so.4
   isn't in the path. Our capture code works around this, but if you want to
   run py_retrieve directly make sure you have
   
        export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

   (this is temporary, see https://jpl.slack.com/archives/CVBUUE5T5/p1664476320620079)

3. Run 

        ./muses_capture setup-targets <instrument>

   to create files needed by
   capture-run (or run-retrieval). This goes into the output directory.
   
   Use an instrument "all" to run all the instruments.
   
4. Run

        ./muses_capture capture-run <instrument>
		
   This generates various pickle/save file for the different steps of the
   processing, that can then be used by the other repositories for testing.
   This updates the files in omi_sounding_1 and tropomi_sounding_1 (we
   currently have just one sounding, but we might potentially want to have
   unit tests for multiple soundings).

   Use an instrument "all" to run all the instruments.
   
5. Run

        ./muses_capture capture-test-data --number-cpu 10
		
   This runs all the capture tests in our pytests in refractor-muses,
   which capture lower level test data (e.g., calls to residual_fm_jacobian).
   This use the top level runs captured in the previous step and runs
   pieces of the retrieval to capture things that we can test in isolation
   without needing to run the full retrieval. You can also do this 
   directly with py-test as described below, it is just a convenience
   to run in muses_capture so you can do this the same way as capture-run

6. Can optionally run the full retrieval by

        ./muses_capture run-retrieval <instrument>

   I'm not sure how useful this is any longer. We had this in place when
   we initially developed the capture, and it might be nice to be able to
   to look at a full run some times to see the output. So we've left this
   in place for now.

All files will be locate in the output/ sub directory of this directory.

Currently (7/2023) the instruments we have are:

- omi
- tropomi
- cris_tropomi
- air_omi

You can use the name "all" to run for all instruments.

