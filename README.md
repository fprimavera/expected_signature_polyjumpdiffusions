# Expected signature of polynomial jump-diffusions in d dimensions

Companion code for the paper "Entire sig-jump-diffusions, their expected signature and applications to
score-based generative modeling" by Christa Cuchiero, Francesca Primavera, and Sara Svaluto-Ferro.

For a `d`-dimensional polynomial jump-diffusion `X`, the expected signature truncated at level `len_max`
satisfies

```
E[Sig(X)_T^{≤ len_max}] = exp(T Lᵀ) ε_∅
```

where `L` is the matrix representation of `Sig(X)_T^{≤ len_max}`'s generator, acting on the flat basis
of words of length `0, ..., len_max` over an alphabet of `d` letters — a space of dimension `sum_k d**k`.

This repository computes the expected signature of polynomial jump-diffusions in closed
form, by assembling the matrix `L` from a table of tensor coefficients of `X`'s
characteristics, and exponentiating it.

## Installation

To use `poly_jump_signature` as a library, pip-install it directly from GitHub:

```bash
pip install git+https://github.com/fprimavera/expected_signature_polyjumpdiffusions.git
```

To run the notebooks or the test suite, clone the repo instead and install the
development dependencies:

```bash
git clone https://github.com/fprimavera/expected_signature_polyjumpdiffusions.git
cd expected_signature_polyjumpdiffusions
pip install -e ".[notebooks,dev]"
```

`iisignature` (a notebook dependency) builds from source and requires numpy to be
present already, so if the install fails, retry with
`pip install iisignature --no-build-isolation`.

## Usage

```python
from poly_jump_signature import (
    create_storage_df, export_template_df, load_storage_from_template_df,
    expected_signature, fromwordtonum,
)

d, max_rank = 2, 3     # state dimension, highest tensor rank with a nonzero coefficient

# 1. coefficient template, placeholder 1.0 everywhere
storage = create_storage_df(max_rank, d)
export_template_df(storage, "coefficients_charact.csv")

# 2. fill the CSV in by hand with the model's values, then reload it
#    (or write the values straight into `storage`, as the notebooks do)
# storage = load_storage_from_template_df("coefficients_charact_filled.csv")

# 3. expected signature at time T, truncated at level len_max
Es = expected_signature(storage, len_max=2, d=d, T=1.0)
```

`Es[fromwordtonum(I, d)]` is the expected `I`-signature component. The coefficient
table must cover every tensor rank up to `len_max`, or `matrix_generator` raises
`ValueError`.

### The coefficient table

One row per `(Tensor_rank, Component_tensor_Q^rank, Order_derivative, Derivative_multiindex)`,
with `Value` the corresponding partial derivative evaluated at `X_0`. Two conventions:

- the tensors `Q^(r)` are symmetric, so components are listed only up to permutation —
  there is a row for `(1,2)` but none for `(2,1)`;
- `Order_derivative` equals `len(Derivative_multiindex)`. Derivatives suffice because a
  rank-`r` component is a polynomial of degree at most `r`, so its Taylor expansion at
  `X_0` terminates at order `r`.

## Package layout

```
poly_jump_signature/
    words.py             word <-> integer index bijection
    shuffle.py           shuffle product of words
    storage.py           coefficient table: construction, CSV I/O, tensor packing
    matrix.py            assembles the generator matrix L
    exp_sig_formula.py   expected signature via matrix exponentiation
tests/
    test_basic.py        pytest suite
notebooks/
    construction_matrix_demo.ipynb                  how L is built, on a minimal example
    expected_signature_poly_jump_diffusions.ipynb   worked model, with a Monte Carlo check
    coefficients_charact.csv                        coefficient template (placeholder 1.0)
    coefficients_charact_filled.csv                 filled-in example, loaded by the demo
```

All logic lives in `poly_jump_signature/`; the notebooks only demonstrate usage.
They assume the working directory is `notebooks/` — `.vscode/settings.json` sets
`jupyter.notebookFileRoot` accordingly.

## Tests

```bash
pytest -q
```

Round-trip correctness of the word/integer bijection, combinatorial identities for the
shuffle product and multiindex counts, rank coverage of `derivatives_to_tensor_df`,
shape and structural checks on `matrix_generator`, and `expected_signature` against the
closed-form moments of a Poisson process.

## Citation

If you use this code, please cite the paper:

```
@misc{cuchiero2026entiresigjumpdiffusionsexpectedsignature,
      title={Entire sig-jump-diffusions, their expected signature and applications to score-based generative modeling}, 
      author={Christa Cuchiero and Francesca Primavera and Sara Svaluto-Ferro},
      year={2026},
      eprint={2611.16995},
      archivePrefix={arXiv},
      primaryClass={math.PR},
      url={https://arxiv.org/abs/2611.16995}, 
}
```
