"""
Title	: rmsd_two_var.py
To Run	: from rmsd_two_var import rmsd_two_var
Author	: Frank Werner
Date	: 20230112
Modf	: n/a

"""

# Import python numerical
import numpy as np


def rmsd_two_var(dataset_1: np.ndarray, dataset_2: np.ndarray) -> np.ndarray:
    # ========================================================
    # Root mean square deviation between two data sets
    #
    # Parameters
    # ---------
    # dataset_1 : float array; first data set
    # dataset_2 : float array; second data set
    #
    # Returns
    # -------
    # rmsd : float; root mean square deviation
    # ========================================================

    isnotnan = (np.isfinite(dataset_1) & np.isfinite(dataset_2)).sum()

    rmsd = (np.array(np.nansum((dataset_1 - dataset_2) ** 2.0) / isnotnan)) ** 0.5

    return rmsd
