"""
Coefficient storage: building, exporting/importing, and converting
to the flat tensor basis the coefficients (values of the partial derivatives
evaluated at X_0) of the components of the symmetric tensors Q^(1), ..., Q^(max_rank) 
that define the generator.

"""
import ast
import itertools
from typing import List, Tuple

import numpy as np
import pandas as pd

from .words import fromlentonum, fromwordtonum


def components_charact_multiindices(i: int, d: int) -> List[Tuple[int, ...]]:
    """
    All multiindices of length i over the alphabet {1, ..., d}, up
    to permutation (i.e. combinations with replacement). 
    
    Observe that:
        -each tensor Q^(1), ..., Q^(max_rank) is symmetric, so its components are 
            indexed by these (unordered) multiindices; 
        -these multiindices are in 1:1 correspondence with the derivative multiindices 
            of order i in d variables.

    Parameters
    ----------
    i : int
        Multiindex length / derivative order (>= 0).
    d : int
        Number of variables / alphabet size.

    Returns
    -------
    List[Tuple[int, ...]], length C(i + d - 1, i)
        Sorted multiindices, each a tuple of length i with entries
        in {1, ..., d}.
    """
    return list(itertools.combinations_with_replacement(range(1, d + 1), i))


def create_storage_df(max_rank: int, d: int) -> pd.DataFrame:
    """
    Build an empty (placeholder-filled) coefficient storage table
    for the components of the tensors Q^(1), ..., Q^(max_rank).

    Parameters
    ----------
    max_rank : int
        Highest tensor rank to generate storage for (>= 1).
    d : int
        Number of variables / alphabet size.

    Returns
    -------
    pd.DataFrame
        One row per (tensor rank, tensor component, derivative
        order, derivative component) combination, with columns:

        - Tensor_rank : int
            Rank r of the tensor Q^(r), in {1, ..., max_rank}.
        - Component_tensor_Q^rank : tuple[int, ...] of length r
            The corresponding component of the tensor Q^(r).
        - Order_derivative : int
            Derivative order, in {0, ..., r}.
        - Derivative_multiindex : tuple[int, ...] of length Order_derivative
            The multiindex representing the partial derivative components.
        - Value : float
            Placeholder 1.0, to be filled in by the user (e.g. via a CSV template) 
            with the true value of the derivative evaluated at X_0.
    """
    rows = []
    for rank in range(1, max_rank + 1):
        for multiindex in components_charact_multiindices(rank, d):
            for order in range(rank + 1):
                for der_multiindex in components_charact_multiindices(order, d):
                    rows.append({
                        'Tensor_rank': rank,
                        'Component_tensor_Q^rank': multiindex,
                        'Order_derivative': order,
                        'Derivative_multiindex': der_multiindex,
                        'Value': 1.0,
                    })
    return pd.DataFrame(rows)


def export_template_df(storage: pd.DataFrame, filename: str = 'coefficients_charact.csv') -> None:
    """
    Write a coefficient storage table to a CSV template, to be
    filled in by hand (e.g. in a spreadsheet) with the actual
    numerical values of the model before being read back with
    `load_storage_from_template_df`.

    Parameters
    ----------
    storage : pd.DataFrame
        As produced by `create_storage_df`.
    filename : str
        Output CSV path.

    Returns
    -------
    None
    """
    storage.to_csv(filename, index=False)


def load_storage_from_template_df(filename: str = 'coefficients_charact_filled.csv') -> pd.DataFrame:
    """
    Read a filled-in CSV template back into a coefficient storage
    DataFrame, parsing the tuple-valued columns (written as e.g.
    "(1, 1)") back into Python tuples.

    Parameters
    ----------
    filename : str
        Input CSV path, with the same columns as produced by
        `create_storage_df` / `export_template_df`.

    Returns
    -------
    pd.DataFrame
        Same schema as `create_storage_df`.
    """
    return pd.read_csv(filename, converters={
        'Component_tensor_Q^rank': ast.literal_eval,
        'Derivative_multiindex': ast.literal_eval,
    })


def derivatives_to_tensor_df(storage: pd.DataFrame, d: int) -> pd.DataFrame:
    """
    Convert a derivative-based coefficient storage table (such as
    the one produced by `create_storage_df`) into a tensor-based
    coefficient storage table.

    Each components of the tensors  Q^(1), ..., Q^(max_rank), 
    originally expressed via its partial derivatives at X_0, 
    is re-expressed as a vector in the flat basis of tensor multi-indices.
    
    

    Parameters
    ----------
    storage : pd.DataFrame
        As produced by `create_storage_df` (or
        `load_storage_from_template_df`). Must contain, for every
        rank in {1, ..., storage['Tensor_rank'].max()}, all of its
        tensor components (see `components_charact_multiindices`).
    d : int
        Number of variables / alphabet size.

    Returns
    -------
    pd.DataFrame
        One row per (Tensor_rank, Component_tensor_Q^rank) pair,
        columns:

        - Tensor_rank : int, in {1, ..., storage['Tensor_rank'].max()}.
        - Component_tensor_Q^rank : tuple[int, ...], the tensor
          component this row belongs to.
        - Vector_tensor_num : np.ndarray, shape
          (fromlentonum(Tensor_rank, d),), dtype float64.
          Vector representation of the tensor component in the flat basis of tensor multi-indices
    """
    max_rank = storage['Tensor_rank'].max()
    rows = []
    grouped = storage.groupby(['Tensor_rank', 'Component_tensor_Q^rank'])
    # rank ranges over 1..max_rank inclusive, matching create_storage_df;
    # rank 0 does not appear in storage and is intentionally skipped.
    for rank in range(1, max_rank + 1):
        for entry in components_charact_multiindices(rank, d):
            vector_tensor = np.zeros(fromlentonum(rank, d))
            if (rank, entry) in grouped.groups:
                subset = grouped.get_group((rank, entry))
                for row in subset.itertuples(index=False):
                    for perm in set(itertools.permutations(row.Derivative_multiindex)):
                        index = fromwordtonum(perm, d)
                        vector_tensor[index] = row.Value
            rows.append({'Tensor_rank': rank, 'Component_tensor_Q^rank': entry, 'Vector_tensor_num': vector_tensor})

    return pd.DataFrame(rows)
