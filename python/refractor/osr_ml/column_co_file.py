from .templated_output import TemplatedOutput
import os

class ColumnCoFile:
    '''We might try to get a general file rather than hardcoding CO, but start
    simply for now.'''
    def __init__(self, output_filename: str | os.PathLike[str], pspec: str | os.PathLike[str]) -> None:
        self.output = TemplatedOutput(pspec, output_filename)

__all__ = ["ColumnCoFile",]        
        
