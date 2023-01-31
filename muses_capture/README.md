Pipeline
========

The MUSE's pipeline is great for running large number of end to end soundings.
But we want to extract out small pieces of this for local system and unit
testing of ReFRACTor.

We have separate code in each of the instrument repositories
(e.g., omi, tropomi) for extracting out the capture tests. I'm not 100%
sure this is right division here, but it is at least what we do for now.
So MUSES pipepline stuff gets handled here, but instrument specific stuff
gets handled in each of the instrument repositories.

This directory uses MUSES's amuse-me tool to setup and run the target
used for capturing data used in ReFRACtor unit testing. This runs the
full pipeline.

The output of the run goes into the directory "output". This will be
local to your directory, it doesn't get checked into git. This directory
is then used by the various capture tests mentioned below (and the
"capture" data does get checked into git.

To run:

1. First set up the MUSES execution environment per the MUSES instructions
   (e.g., use build_supplement, or run build.sh and 
   build-python-programs.sh in amuse-me - both are needed).

2. Make sure amuse-me in on your path (e.g.
   export PATH=/home/smyth/muses/amuse-me/bin:$PATH )

3. Make muses-vlidort (not currently part of build by amuse me)
   
        git clone git@github.jpl.nasa.gov:MUSES-Processing/muses-vlidort.git
		cd muses-vlidort/build/release
		cmake -DCMAKE_BUILD_TYPE=Release ../..
		make

4. While omi runs use the new muses-vlidort,  tropomi still uses a version
   in the OSP directory. For this one, there is a bug for libgfortran.so.4
   isn't in the path. Our capture code works around this, but if you want to
   run py_retrieve directly make sure you have
   
        export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

   (this is temporary, see https://jpl.slack.com/archives/CVBUUE5T5/p1664476320620079)

5. Run 

        ./muses_capture setup-targets <instrument>

   to create files needed by
   capture-run (or run-retrieval). This goes into the output directory.
   
6. Run

        ./muses_capture capture-run <instrument>
		
	This generates various pickle/save file for the different steps of the
	processing, that can then be used by the other repositories for testing.
	This updates the files in omi_sounding_1 and tropomi_sounding_1 (we
	currently have just one sounding, but we might potentially want to have
    unit tests for multiple soundings).

7. Can optionally run the full retrieval by

        ./muses_capture run-retrieval <instrument>

   I'm not sure how useful this is any longer. We had this in place when
   we initially developed the capture, and it might be nice to be able to
   to look at a full run some times to see the output. So we've left this
   in place for now.

All files will be locate in the output/ sub directory of this directory.

Currently (1/2023) the instruments we have are:

- omi
- tropomi

Once these two scripts have finished you can run capture tests for each of
the instruments. We preface all the capture tests with "test_capture",
so you can run the full set (in the instrument repository) with 

    pytest -rxXs test/ -k test_capture --run-capture

You can see all tests (without running them) by

    pytest --collect-only test/*_test.py -k test_capture --run-capture

OMI
---

Individual tests (as of 1/2023, the list might grow).

    pytest -rxXs test/retrieval_test.py -k test_capture_uip[1] --run-capture
    pytest -rxXs test/retrieval_test.py -k test_capture_uip[2] --run-capture
    pytest -rxXs test/retrieval_test.py -k test_capture_retrieval_data[1] --run-capture
    pytest -rxXs test/retrieval_test.py -k test_capture_retrieval_data[2] --run-capture
    pytest -rxXs test/refractor_fm_Test.py -k test_capture_fm_wrapper[1] --run-capture
    pytest -rxXs test/refractor_fm_Test.py -k test_capture_fm_wrapper[2] --run-capture
    pytest -rxXs test/radiance_test.py -k test_capture_atmposphere --run-capture
    pytest -rxXs test/radiance_test.py -k test_capture_expected_xsec --run-capture
