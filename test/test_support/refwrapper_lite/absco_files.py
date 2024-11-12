from pathlib import Path
import refractor.framework as rf

def absco_file_list(osp_dir: str):
    osp_dir = Path(osp_dir)
    return [
        ('co', (osp_dir / 'ABSCO' / 'v1.0_SWIR_CO' / 'nc_ABSCO' / 'CO_04124-04385_v0.0_init.nc').as_posix()),
        ('ch4', (osp_dir / 'ABSCO' / 'v1.0_SWIR_CH4' / 'nc_ABSCO' / 'CH4_04124-04385_v0.0_init.nc').as_posix()),
        ('co', (osp_dir / 'ABSCO' / 'v1.0_SWIR_H2O' / 'nc_ABSCO' / 'H2O_04124-04385_v0.0_init.nc').as_posix()),
        ('co', (osp_dir / 'ABSCO' / 'v1.0_SWIR_HDO' / 'nc_ABSCO' / 'HDO_04124-04385_v0.0_init.nc').as_posix()),
    ]

class AbscoStub(rf.Absco):
    # JLL: this might be obsolete, this was something I got from an early example notebook

    def __init__(self, proxied_obj):
        # Always call the super class's constructor to initialize the director relationship
        super().__init__()

        if not isinstance(proxied_obj, rf.Absco):
            raise TypeError("proxied_obj is not of type rf.Absco")

        self.proxied = proxied_obj

    def table_scale(self, wn: float) -> float:
        return self.proxied.table_scale(wn)

    def broadener_vmr_grid(self, broadener_index: int) -> rf.BlitzArray_double_1:
        return self.proxied.broadener_vmr_grid(broadener_index)

    # In SWIG this method is defined as a %python_attribute
    # In this director we must implement the underlying method
    # that was renamed in order for other C++ classes to be able
    # to use our functinality
    def _v_pressure_grid(self) -> rf.BlitzArray_double_1:
       # Here we can still use the Python attribute version of the C++ method
       return self.proxied.pressure_grid

    # In SWIG this method is defined as a %python_attribute
    # In this director we must implement the underlying method
    # that was renamed in order for other C++ classes to be able
    # to use our functinality
    def _v_temperature_grid(self) -> rf.BlitzArray_double_2:
       # Here we can still use the Python attribute version of the C++ method
       return self.proxied.temperature_grid

    # Tells ABSCO code native format of underlying data file to reduce conversion bottlenecks
    def is_float(self) -> bool:
        return self.proxied.is_float()

    # double in the method name refers to the C++ double type
    def read_double(self, wn: float) -> rf.BlitzArray_double_3:
        return self.proxied.read_double(wn)

    # double in the method name refers to the C++ double type
    def read_float(self, wn: float) -> rf.BlitzArray_float_3:
        return self.proxied.read_float(wn)

    ####
    # The following come from GasAbsorption's abstract interface

    def have_data(self, wn: float) -> bool:
        return self.proxied.have_data(wn)

    # In SWIG numer_broadener is defined as a %python_attribute
    # In this director we must implement the underlying method
    # that was renamed in order for other C++ classes to be able
    # to use our functinality
    def _v_number_broadener(self) -> int:
        # Here we can still use the Python attribute version of the C++ method
        return self.proxied.number_broadener

    def broadener_name(self, broadener_index: int) -> str:
        return self.proxied.broadener_name(broadener_index)

    # This method is overloaded.
    # The output class type should be selected based upon the input types:
    # - temp: DoubleWithUnit and broadener_vmr: ArrayWithUnit -> DoubleWithUnit
    # - temp: AutoDerivativeWithUnitDouble, broadener_vmr: ArrayAdWithUnit_double_1 -> AutoDerivativeWithUnitDouble
    def absorption_cross_section(self, wn: float, press: rf.DoubleWithUnit, temp, broadener_vmr):
        # If changing the implementation you need to be aware of the types as indicated above
        # But if we are just passing along values then SWIG will do that determination for us
        return self.proxied.absorption_cross_section(wn, press, temp, broadener_vmr)