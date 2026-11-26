"""
Shuffle product of words.
"""
from typing import List

import numpy as np

from .words import fromlentonum, fromwordtonum


def shuffle(word1: List[int], word2: List[int]) -> List[List[int]]:
    """
    Compute the shuffle product of two words (multi-indices).

    Parameters
    ----------
    word1 : List[int], shape (p,)
        First word.
    word2 : List[int], shape (q,)
        Second word.

    Returns
    -------
    List[List[int]]
        List of words, each of length p+q.
    """
    sh = []
    if len(word1) == 0:
        return [word2]
    if len(word2) == 0:
        return [word1]

    for p in shuffle(word1[:-1], word2):
        sh.append(p + [word1[-1]])
    for p in shuffle(word1, word2[:-1]):
        sh.append(p + [word2[-1]])
    return sh


def shuffle_num(word1: List[int], word2: List[int], d: int) -> np.ndarray:
    """

    Vector representation of the shuffle product of
    two words in the flat basis of tensor multi-indices

    Parameters
    ----------
    word1 : List[int], shape (p,)
        First word.
    word2 : List[int], shape (q,)
        Second word.
    d : int
        Alphabet size.

    Returns
    -------
    np.ndarray, shape (fromlentonum(p + q, d),), dtype float64
        Vector of counts: the entry at position `fromwordtonum(w, d)`
        is the number of ways to interleave word1 and word2 that give
        word w. Since every resulting word has length p + q, the
        vector only needs to be long enough to cover words up to that
        length.
    """
    shuffle_words = shuffle(word1, word2)
    vec_dim = fromlentonum(len(word1) + len(word2), d)

    shuffle_num_vector = np.zeros(vec_dim)
    for w in shuffle_words:
        num = fromwordtonum(w, d)
        shuffle_num_vector[num] += 1
    return shuffle_num_vector
