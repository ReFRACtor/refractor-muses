import refractor.muses.muses_py as mpy
import collections.abc
import re
import pandas as pd
import io
from functools import lru_cache

class TesFile(collections.abc.Mapping):
    '''There are a number of files that are in the "TES File" format. This is made
    up of a header with keyword/value pairs and a (possibly empty) table.

    There is muses-py code for reading this, but this is really pretty straight forward
    so we just implement this our self. We can turn around and use the muse-py version if
    we run into any issue.

    We present the various keyword/value pairs as dictionary like interface.

    The attribute table is either None or a pandas data frame with the table content
    '''
    def __init__(self, fname : str, use_mpy=False):
        '''Open the given file, and read the keyword/value pairs plus the
        (possibly empty) table.

        Note that you generally shouldn't call this initializer, rather use
        TesFile.create which adds caching, so opening the same file twice returns
        the same object.

        As a convenience for testing, you can specify use_mpy as True to use the
        old mpy code. This may go away at some point, but for now it is useful to
        test that we implement the reading correctly.
        '''
        self.file_name = fname
        if(use_mpy):
            _, d = mpy.read_all_tes(fname)
            self.mpy_d = d
            self._d = d['preferences']
            if(d['numRows'] > 0):
                tbl = f"{d['labels1']}\n" + "\n".join(d['data'])
                self.table = pd.read_table(io.StringIO(tbl),sep=r'\s+', header=0)
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
            # The table has 2 header lines, the actual header and the units.
            # pandas can actually create a table like this, using a multindex.
            # But this is more complicate they we want, so just split out the
            # second header and treat it separately
            t = tbl.lstrip().splitlines()
            hdr = t[0]
            self.table_units = t[1].split()
            body = "\n".join(t[2:])
            # Determine number of rows. The file may have extra lines at the bottom,
            # so we make sure to only read the number of rows that the file claims
            # is there.
            self.table = pd.read_table(io.StringIO(f"{hdr}\n{body}"),sep=r'\s+', header=0,
                                       nrows=self.shape[0])
        else:
            self.table = None

    @property
    def shape(self):
        '''Return the shape of the table. Note this comes from the metadata 'Data_Size',
        not the actual table.'''
        return [int(i) for i in self["Data_Size"].split('x')]
            
    def __getitem__(self, ky):
        return self._d[ky]

    def __len__(self):
        return len(self._d)

    def __iter__(self):
        return self._d.__iter__()

    @classmethod
    @lru_cache(maxsize=50)
    def create(cls, fname : str):
        '''This creates a TesFile that reads the given file. Because we often
        open the same file multiple times in different contexts, this adds caching so
        open the same file a second time just returns the existing TesFile object.
        '''
        return cls(fname)

__all__ = ["TesFile", ]
