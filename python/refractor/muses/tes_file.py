import refractor.muses.muses_py as mpy
import collections.abc
import re
import pandas as pd
import io

class TesFile(collections.abc.Mapping):
    '''There are a number of files that are in the "TES File" format. This is made
    up of a header with keyword/value pairs and a (possibly empty) table.

    There is muses-py code for reading this, but this is really pretty straight forward
    so we just implement this our self. We can turn around and use the muse-py version if
    we run into any issue.

    We present the various keyword/value pairs as dictionary like interface.

    The attribute table is either None or a pandas data frame with the table content
    '''
    def __init__(self, fname, use_mpy=False):
        self.file_name = fname
        if(use_mpy):
            _, d = mpy.read_all_tes(fname)
            self.mpy_d = d
            self._d = d['preferences']
            if(d['numRows'] > 0):
                tbl = f"{d['labels1']}\n{d['labels2']}\n" + "\n".join(d['data'])
                self.table = pd.read_table(io.StringIO(tbl),sep=r'\s+', header=[0,1])
            else:
                self.table = None
            return

        t = open(fname).read()
        
        # Make sure we find the end of the header
        if(not re.search(r'^\s*end_of_header.*$', t, flags=re.MULTILINE | re.IGNORECASE)):
            raise RuntimeError(f"Didn't find end_of_header in file {self.file_name}")
        
        # Split into header and table part. Note not all files have table part, but
        # should have a header part
        t2 = re.split(r'^\s*end_of_header.*$', t, flags=re.MULTILINE | re.IGNORECASE)
        if(len(t2) not in (1,2)):
            raise RuntimeError(f"Trouble parsing file {self.file_name}")
        hdr = t2[0]
        tbl = t2[1] if len(t2) == 2 else None
        
        # Strip comments out
        hdr = re.sub(r'//(.*)$', '', hdr, flags=re.MULTILINE)

        # Process each line in header and fill in keyword=value data
        self._d = {}
        for ln in re.split(r'\n', hdr):
            m = re.match(r'\s*(\S*)\s*=\s*(.*\S*)\s*', ln)
            if(m):               # Just skip lines that don't have keyword=value form
                # Strip off any quotes
                self._d[m[1]] = re.sub(r'"', '', m[2]).strip()

        if(tbl is not None and re.search(r'\S', tbl)):
            self.table = pd.read_table(io.StringIO(tbl),sep=r'\s+', header=[0,1])
        else:
            self.table = None
            
    def __getitem__(self, ky):
        return self._d[ky]

    def __len__(self):
        return len(self._d)

    def __iter__(self):
        return self._d.__iter__()

__all__ = ["TesFile", ]
