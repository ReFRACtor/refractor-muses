from __future__ import annotations
import refractor.framework as rf  # type: ignore
import numpy as np

# Most of our various pieces used in the forward model have C++ base class (e.g.,
# Temperature. There is a natural way to make a python version by deriving from
# a class like TemperatureImpBase. However, we may run into completely new
# objects. These can of course be added to C++ framework code, but it can be useful
# to have a all python implementation. This can be useful for initial development,
# and even used long term if there is no need to call C++ (e.g., a forward model all
# in python).
#
# This is a basic test of that, adding a "PythonExampleState" object attached to a
# state vector.


class PythonExampleState(rf.GenericStateImpBase):
    def __init__(self, val: np.ndarray, mp: rf.StateMapping):
        super().__init__()
        self.init(val, mp)

    def desc(self):
        return "PythonExampleState"

    def clone(self) -> rf.GenericState:
        return PythonExampleState(self.mapped_state.value, self.state_mapping)

    def sub_state_identifier(self) -> str:
        return "python_example_state"

    def state_vector_name_i(self, i: int) -> str:
        mname = self.state_mapping.name
        if mname != "linear":
            return f"{mname} Python Example State {i + 1}"
        f"Python Example State {i + 1}"

    @property
    def my_state(self) -> rf.ArrayAd_double_1:
        return self.mapped_state


def test_python_example_state():
    psv = PythonExampleState([1, 2, 3], rf.StateMappingLog())
    fm_sv = rf.StateVector()
    fm_sv.add_observer(psv)
    fm_sv.update_state(psv.coefficient.value)
    print(fm_sv)
    print(psv.my_state)
    fm_sv.update_state([2, 3, 4])
    print(fm_sv)
    print(psv.my_state)
