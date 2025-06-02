from __future__ import annotations
from refractor.muses import RecordAndPlayFunc

class SimpleFunc:
    def __init__(self, rplay : RecordAndPlayFunc | None = None) -> None:
        self.val = 0
        self._rplay = rplay

    def add(self, v : float) -> None:
        if(self._rplay is not None):
            self._rplay.record("add", v)
        self.val += v

    def add_mul(self, v : float, v2: float) -> None:
        if(self._rplay is not None):
            self._rplay.record("add_mul", v, RecordAndPlayFunc.PlaceHolder("v2"))
        self.val += v * v2

def test_rplay():
    p = RecordAndPlayFunc()
    sfunc = SimpleFunc(p)
    sfunc.add(10.0)
    sfunc.add(20.0)
    sfunc.add_mul(30.0, 2)
    assert sfunc.val == 10+20+30*2
    sfunc2 = SimpleFunc()
    p.play(sfunc2, {'v2' : 3})
    assert sfunc2.val == 10+20+30*3
    
           
    
        
