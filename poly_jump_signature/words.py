"""
Word-indexing utilities.

Bijection between finite words (multi-indices) over the alphabet
{1, ..., d} and non-negative integers, using a bijective base-d
numeral system. Word length 0 (the empty word) maps to 0; the d
words of length 1 map to 1..d; the d**2 words of length 2 map to
d+1 .. d+d**2; and so on. This matches the flat tensor-basis
ordering used throughout the rest of the package.

"""
from typing import List, Sequence

import numpy as np


def fromwordtonum(I: Sequence[int], d: int) -> int:
    """
    Encode a word as its integer index in the bijective base-d
    numbering.

    Parameters
    ----------
    I : Sequence[int], shape (L,)
        Word (multi-index): a sequence of L letters, each in
        {1, ..., d}. L = 0 (empty sequence) is allowed.
    d : int
        Alphabet size (a positive integer).

    Returns
    -------
    int
        Non-negative integer index of the word. 0 iff I is empty.

    Raises
    ------
    ValueError
        If any entry of I is not in {1, ..., d}.
    """
    I = np.asarray(I)
    if I.size == 0:
        return 0
    if np.any((I < 1) | (I > d)):
        raise ValueError(f"all entries of I must be in {{1,...,{d}}}, got {I}")
    L = I.size
    powers = d ** np.arange(L - 1, -1, -1)
    return int(np.dot(I, powers))


def fromnumtoword(n: int, d: int) -> List[int]:
    """
    Decode an integer index back into a word (inverse of
    `fromwordtonum`).

    Parameters
    ----------
    n : int
        Non-negative integer index, as produced by `fromwordtonum`.
    d : int
        Alphabet size (a positive integer).

    Returns
    -------
    List[int]
        Word (multi-index) of variable length, each entry in
        {1, ..., d}. Empty list iff n == 0.
    """
    if n == 0:
        return []
    return fromnumtoword((n - 1) // d, d) + [(n - 1) % d + 1]


def fromlentonum(len_max: int, d: int) -> int:
    """
    Dimension of the flat vector space spanned by all words of
    length 0, 1, ..., len_max over an alphabet of size d.


    Parameters
    ----------
    len_max : int
        Maximum word length to include (>= 0).
    d : int
        Alphabet size (a positive integer).

    Returns
    -------
    int
        Dimension of the corresponding flat vector space (>= 1).
    """
    maxword = d * np.ones(len_max)
    return fromwordtonum(maxword, d) + 1
