"""
Test suite for the poly_jump_signature package.

Run with:  pytest -q
"""
import itertools
import math

import numpy as np
import pytest
from scipy.linalg import expm

from poly_jump_signature import (
    fromwordtonum, fromnumtoword, fromlentonum,
    shuffle, shuffle_num,
    components_charact_multiindices, create_storage_df,
    derivatives_to_tensor_df, matrix_generator, expected_signature,
)


# ---------------------------------------------------------------------------
# Word indexing
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("d", [1, 2, 3])
@pytest.mark.parametrize("n", range(0, 40))
def test_num_to_word_to_num_roundtrip(n, d):
    word = fromnumtoword(n, d)
    assert fromwordtonum(word, d) == n


@pytest.mark.parametrize("d", [1, 2, 3])
@pytest.mark.parametrize("length", [0, 1, 2, 3])
def test_word_to_num_to_word_roundtrip(length, d):
    for word in itertools.product(range(1, d + 1), repeat=length):
        word = list(word)
        n = fromwordtonum(word, d)
        assert fromnumtoword(n, d) == word


def test_fromwordtonum_rejects_out_of_range_letters():
    with pytest.raises(ValueError):
        fromwordtonum([1, 5], d=2)


@pytest.mark.parametrize("d", [1, 2, 3])
@pytest.mark.parametrize("len_max", [0, 1, 2, 3])
def test_fromlentonum_matches_geometric_sum(len_max, d):
    expected = sum(d ** k for k in range(len_max + 1))
    assert fromlentonum(len_max, d) == expected


@pytest.mark.parametrize("d", [1, 2, 3])
@pytest.mark.parametrize("len_max", [0, 1, 2, 3])
def test_fromlentonum_is_word_count_up_to_length(len_max, d):
    # fromlentonum(len_max, d) must equal the number of distinct words
    # of length 0..len_max, i.e. the size of the flat index range
    # [0, fromlentonum(len_max, d) - 1].
    count = sum(d ** k for k in range(len_max + 1))
    assert fromlentonum(len_max, d) == count


# ---------------------------------------------------------------------------
# Shuffle
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("p,q", [(0, 0), (1, 0), (0, 2), (1, 1), (2, 1), (2, 2)])
def test_shuffle_count_matches_binomial(p, q):
    a = list(range(1, p + 1))
    b = list(range(p + 1, p + q + 1))  # disjoint letters -> all interleavings distinct
    result = shuffle(a, b)
    assert len(result) == math.comb(p + q, p)


@pytest.mark.parametrize("p,q", [(1, 1), (2, 1), (2, 2), (3, 1)])
def test_shuffle_num_single_letter_alphabet_is_binomial(p, q):
    # With d=1 every word of length L is [1]*L, so all C(p+q, p)
    # interleavings collapse onto the single word of length p+q.
    d = 1
    word1 = [1] * p
    word2 = [1] * q
    vec = shuffle_num(word1, word2, d)
    assert vec.shape == (fromlentonum(p + q, d),)
    expected = np.zeros_like(vec)
    expected[fromwordtonum([1] * (p + q), d)] = math.comb(p + q, p)
    np.testing.assert_array_equal(vec, expected)


def test_shuffle_num_total_count_matches_binomial():
    d = 2
    word1, word2 = [1, 2], [2]
    vec = shuffle_num(word1, word2, d)
    assert vec.sum() == math.comb(len(word1) + len(word2), len(word1))


# ---------------------------------------------------------------------------
# Multiindices / storage
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("d", [1, 2, 3])
@pytest.mark.parametrize("i", [0, 1, 2, 3])
def test_components_charact_multiindices_count(i, d):
    result = components_charact_multiindices(i, d)
    assert len(result) == math.comb(i + d - 1, i)
    # all sorted / non-decreasing (combinations, no permutations)
    assert all(list(t) == sorted(t) for t in result)


def test_derivatives_to_tensor_df_covers_all_ranks():
    d, max_rank = 2, 3
    storage = create_storage_df(max_rank, d)
    df = derivatives_to_tensor_df(storage, d)
    assert set(df["Tensor_rank"].unique()) == set(range(1, max_rank + 1))
    # one row per (rank, component) pair
    expected_rows = sum(len(components_charact_multiindices(r, d)) for r in range(1, max_rank + 1))
    assert len(df) == expected_rows


# ---------------------------------------------------------------------------
# matrix_generator
# ---------------------------------------------------------------------------

def test_matrix_generator_raises_if_coefficients_incomplete():
    d = 2
    storage = create_storage_df(1, d)  # only rank 1 available
    with pytest.raises(ValueError):
        matrix_generator(storage, len_max=2, d=d)


def test_matrix_generator_shape_and_first_column_zero():
    d = 2
    storage = create_storage_df(3, d)
    len_max = 2
    M = matrix_generator(storage, len_max, d)
    dim_L = fromlentonum(len_max, d)
    assert M.shape == (dim_L, dim_L)
    # column 0 corresponds to the generator applied to the empty word,
    # which is 0 by convention (column n holds the image of the basis
    # word with index n -- see matrix_generator docstring).
    np.testing.assert_array_equal(M[:, 0], np.zeros(dim_L))


def test_matrix_generator_no_nan_or_inf():
    d = 2
    storage = create_storage_df(3, d)
    M = matrix_generator(storage, len_max=2, d=d)
    assert np.isfinite(M).all()


# ---------------------------------------------------------------------------
# expected_signature
# ---------------------------------------------------------------------------

def test_expected_signature_shape():
    d = 2
    len_max = 2
    storage = create_storage_df(3, d)
    result = expected_signature(storage, len_max, d, T=1.0)
    dim_L = fromlentonum(len_max, d)
    assert result.shape == (dim_L,)


def test_expected_signature_at_time_zero_is_first_basis_vector():
    # At T=0, expm(0) = identity, so expected_signature should return e_0 .
    d = 2
    len_max = 2
    storage = create_storage_df(3, d)
    dim_L = fromlentonum(len_max, d)
    result = expected_signature(storage, len_max, d, T=0.0)
    expected = np.zeros(dim_L)
    expected[0] = 1.0
    np.testing.assert_allclose(result, expected, atol=1e-12)


def test_expected_signature_no_nan_or_inf():
    d = 2
    storage = create_storage_df(3, d)
    result = expected_signature(storage, len_max=2, d=d, T=1.0)
    assert np.isfinite(result).all()


def test_expected_signature_matches_poisson_process_known_moments():
    # Regression/correctness check pinned against an independent,
    # exact closed form (not derived from this package): for a pure
    # jump Poisson process N of rate lambda (single coordinate,
    # jump kernel lambda * delta_1, so b = Q^(r) = lambda for every
    # rank r -- the jump kernel's first moment must be folded into
    # b, it is not added separately), each jump of size 1 acts as a
    # straight-line lift contributing a factor exp(z) to the
    # (commutative, 1D) signature generating function. Hence
    # Sig^(k)_T = N_T^k / k!, so E[Sig^(k)_T] = E[N_T^k] / k!, using
    # the raw moments of a Poisson(lambda*T) random variable. If
    # this test starts failing, check the row-vs-column convention
    # in expected_signature first (see its docstring).
    d = 1
    max_rank = 3
    lam = 2.0
    T = 1.0
    mu = lam * T  # Poisson mean

    storage = create_storage_df(max_rank, d)
    storage['Value'] = 0.0
    for r in range(1, max_rank + 1):
        mask = (
            (storage['Tensor_rank'] == r)
            & (storage['Component_tensor_Q^rank'] == (1,) * r)
            & (storage['Order_derivative'] == 0)
        )
        storage.loc[mask, 'Value'] = lam

    result = expected_signature(storage, max_rank, d, T)

    # raw moments of Poisson(mu): E[X]=mu, E[X^2]=mu+mu^2, E[X^3]=mu+3mu^2+mu^3
    expected_level1 = mu
    expected_level2 = (mu + mu**2) / math.factorial(2)
    expected_level3 = (mu + 3 * mu**2 + mu**3) / math.factorial(3)

    idx1 = fromwordtonum((1,), d)
    idx2 = fromwordtonum((1, 1), d)
    idx3 = fromwordtonum((1, 1, 1), d)

    np.testing.assert_allclose(result[idx1], expected_level1)
    np.testing.assert_allclose(result[idx2], expected_level2)
    np.testing.assert_allclose(result[idx3], expected_level3)
