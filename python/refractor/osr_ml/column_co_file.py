from __future__ import annotations
from .templated_output import TemplatedOutput
import os
import numpy as np
from .declaritve_output import register_dataset, DeclarativeOutput
import typing

if typing.TYPE_CHECKING:
    from .ml import MlPredictionClass

class _DimColumn:
    '''Make dim_column a enumeration'''
    def __init__(self):
        self.__self__ = self
        
    def _creator(self, t, ds, var_name) -> None:
        en = ds.createEnumType(np.uint8, 'dim_column', {'Column':0, 'Trop':1,"UpperTrop":2, "LowerTrop" : 3, "Strato" : 4})
        ds.createVariable(var_name, en, ('dim_column',))

    def __call__(self) -> np.ndarray:
        return np.array([0,1,2,3,4], dtype=np.uint8)
    
        
class ColumnCoFile(DeclarativeOutput):
    """We might try to get a general file rather than hardcoding CO, but start
    simply for now."""

    def __init__(
       self, prediction: MlPredictionClass, output_filename: str | os.PathLike[str], pspec: str | os.PathLike[str]
    ) -> None:
        self.prediction = prediction
        self.output = TemplatedOutput(pspec, output_filename)
        self.output.register_instances((self,))
        self.output.register_variable("dim_column", _DimColumn())

    @register_dataset("/col")
    def co_column(self) -> np.ndarray:
        # We get the columns to use from the file labels_order.txt
        return self.prediction.labels_pred[:,:5]

    @register_dataset("/col_error")
    def co_column(self) -> np.ndarray:
        # We get the columns to use from the file labels_order.txt
        return self.prediction.labels_pred[:,-5:]
    
    def write(self):
        self.output.write()


__all__ = [
    "ColumnCoFile",
]
