import os
import re

from refractor import framework as rf
from refractor.framework import refractor_config, creator

from .retrieval_run_config import retrieval_run_config

@refractor_config
def run_config(**kwargs):

    config_def = retrieval_run_config(**kwargs)

    config_def['atmosphere']['absorber']['creator'] = creator.absorber.AbsorberXSec
    config_def['atmosphere']['absorber']['default_gas_definition']['creator'] = creator.absorber.CrossSectionGasDefinition

    return config_def
