import refractor.muses.muses_py as mpy
from contextlib import contextmanager
import tempfile
import os

class MusesObservation:
    '''It isn't clear what exactly we want in this class. The Observation
    in py-retrieve also include other things, like the SpectralDomain,
    bad samples, and various observation metadata like the solar and viewing
    geometry.

    For now, we wrap this up into a class. At least for now, we'll also keep
    the various dictionaries py-retrieve has like o_airs etc.

    We wrap up the existing py-retrieve calls for reading this data.

    We may modify this over time, but this is at least a good place to start.
    '''
    def __init__(self, muses_py_dict, channel_list):
        self.muses_py_dict = muses_py_dict
        self.channel_list = channel_list

    @contextmanager
    def osp_setup(self, osp_dir=None):
        '''Some of the readers assume the OSP is available as "../OSP". We
        are trying to get away from assuming we are in a run directory
        whenever we do things, it limits using the code in various contexts.
        So this handles things by taking the osp_dir and setting up a
        temporary directory so things look like muses_py assumes.

        We can perhaps just move the muses-py code over at some point and
        handle this more cleanly, but for now we do this.'''
        if(osp_dir is None):
            dname = os.path.abspath("../OSP")
        else:
            dname = os.path.abspath(osp_dir)
        curdir = os.path.abspath(os.path.curdir)
        try:
            with tempfile.TemporaryDirectory() as tname:
                os.chdir(tname)
                os.symlink(dname, "OSP")
                os.mkdir("subdir")
                os.chdir("./subdir")
                yield
        finally:
            os.chdir(curdir)
            
        

# muses_forward_model has an older observation class named MusesAirsObservation.
# Short term we add a "New" here. We should sort that out -
# we are somewhat rewriting MusesObservationBase to not use a UIP. This will
# probably get married into one clases, but we aren't ready to do that yet.

class MusesAirsObservationNew(MusesObservation):
    def __init__(self, filename, xtrack, atrack, channel_list, osp_dir=None):
        i_fileid = {}
        i_fileid['preferences'] = {'AIRS_filename' : os.path.abspath(filename),
                                   'AIRS_XTrack_Index' : xtrack,
                                   'AIRS_ATrack_Index' : atrack}
        i_window = []
        for cname in channel_list:
            i_window.append({'filter': cname})
        with(self.osp_setup(osp_dir)):
            o_airs = mpy.read_airs(i_fileid, i_window)
        super().__init__(o_airs, channel_list)
