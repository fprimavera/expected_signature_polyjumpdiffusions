"""
Computation of the expected signature vector, via matrix
exponentiation of the generator matrix L.
"""

import numpy as np
import pandas as pd
from scipy.linalg import expm

from .matrix import matrix_generator


def expected_signature(coeff_df: pd.DataFrame, len_max: int, d: int, T: float) -> np.ndarray:
    """
    Compute the expected signature of the process, truncated at
    level `len_max`, at time T.

    Parameters
    ----------
    coeff_df : pd.DataFrame
        Coefficient storage, as produced by `create_storage_df` (or
        `load_storage_from_template_df`). Must contain coefficients
        for all tensors of rank up to `len_max`.
    len_max : int
        Truncation level (maximum word length considered).
    d : int
        Number of variables / alphabet size.
    T : float
        Time at which to evaluate the expected signature.

    Returns
    -------
    np.ndarray, shape (dim_L,)
        Vector representing E[Sig(X)_T] truncated at level
        `len_max`, where dim_L = fromlentonum(len_max, d). Entry at
        position `fromwordtonum(I, d)` is E[Sig^I(X)_T], the
        expected I-signature component.

    """
    
    matL = matrix_generator(coeff_df, len_max, d)
    return expm(T * matL)[0, :]
