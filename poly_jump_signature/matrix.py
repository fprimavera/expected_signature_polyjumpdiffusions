"""
Assembly of the matrix L representing the action of the generator
on the flat tensor basis, used to compute the expected signature
via matrix exponentiation.
"""
import math

import numpy as np
import pandas as pd

from .storage import derivatives_to_tensor_df
from .words import fromlentonum, fromnumtoword
from .shuffle import shuffle_num


def matrix_generator(coeff_df: pd.DataFrame, len_max: int, d: int) -> np.ndarray:
    """
    Build the matrix L representing the generator's action on the flat basis
    of words of length 0, ..., len_max in an alphabet of letters {1, ..., d}.

    Parameters
    ----------
    coeff_df : pd.DataFrame
        Coefficient storage, as produced by `create_storage_df` (or
        `load_storage_from_template_df`). Must have
        coeff_df['Tensor_rank'].max() >= len_max, i.e. contain
        coefficients for all tensors of rank up to len_max.
    len_max : int
        Maximum word length considered (>= 1); determines the size
        of the returned matrix together with `d`.
    d : int
        Number of variables / alphabet size.

    Returns
    -------
    np.ndarray, shape (dim_L, dim_L), dtype float64
        where dim_L = fromlentonum(len_max, d) (dimension of the
        flat vector space of words of length 0, ..., len_max).
        Column n (0-indexed) holds the image under the generator of
        the basis word with index n (`fromnumtoword(n, d)`); column
        0 corresponds to the empty word and is identically 0 (the
        generator applied to the empty word is 0).

    Raises
    ------
    ValueError
        If coeff_df does not contain coefficients of the tensors
        up to rank len_max.
    AssertionError
        If, for some tensor rank/component pair needed during
        assembly, coeff_df does not have exactly one matching row
        (indicates an incomplete or duplicated coefficient table).
    """
    max_rank = coeff_df['Tensor_rank'].max()
    if max_rank < len_max:
        raise ValueError(
            f"Coefficient matrix incomplete: need tensors up to rank {len_max}, "
            f"but only rank {max_rank} is available.")

    df = derivatives_to_tensor_df(coeff_df, d)
    dim_L = fromlentonum(len_max, d)
    rows = [np.zeros(dim_L)]

    for n in range(1, dim_L):
        word = fromnumtoword(n, d)
        L_col = np.zeros(dim_L)

        for j in range(1, len(word) + 1):
            word_shift = word[:len(word) - j]
            shift = tuple(sorted(word[len(word) - j:]))
            mask = (df["Tensor_rank"] == j) & df["Component_tensor_Q^rank"].apply(lambda x: x == shift)

            matched = df.loc[mask, "Vector_tensor_num"]
            assert len(matched) == 1, f"Expected 1 row for j={j}, shift={shift}, found {len(matched)}"
            coeff_charact = matched.to_numpy()[0]

            indices = np.nonzero(coeff_charact)[0]
            product_tensor_num = np.zeros(dim_L)
            for val in indices:
                base_tensor = fromnumtoword(val, d)
                local_vec = shuffle_num(base_tensor, word_shift, d)
                # local_vec uses the same global fromwordtonum numbering,
                # just sized to its own local max word-length (up to
                # len(base_tensor) + len(word_shift)). Cap against dim_L
                # (the full matrix dimension), not len_max + 1: words of
                # length up to len_max can have an index as large as
                # dim_L - 1.
                take = min(local_vec.size, dim_L)
                product_tensor_num[:take] += coeff_charact[val] * local_vec[:take]
            L_col += product_tensor_num / math.factorial(j)

        rows.append(L_col)

    return np.vstack(rows).T

