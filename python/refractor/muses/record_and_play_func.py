from __future__ import annotations
from typing import Any

class RecordAndPlayFunc:
    '''It can be useful for testing to be able to record the function calls to
    an object, and then play them back.

    It is useful to have some arguments "special" and not actually saved, but rather
    have a placeholder than gets replaced in the call back.'''
    class PlaceHolder:
        def __init__(self, nm : str) -> None:
            self.nm = nm

    def __init__(self) -> None:
        self._record = []

    def record(self, funcname : str, *args : list[Any]) -> None:
        self._record.append([funcname, args])

    def play(self, obj: Any, placeholder : dict[str, Any]) -> None:
        for funcname, avlist in self._record:
            alist = []
            for av in avlist:
                if(isinstance(av, RecordAndPlayFunc.PlaceHolder)):
                    alist.append(placeholder[av.nm])
                else:
                    alist.append(av)
            getattr(obj, funcname)(*alist)
