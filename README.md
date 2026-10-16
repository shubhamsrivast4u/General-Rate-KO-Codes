# HiKO and Neural Permutation Selection

This repository contains the code for the paper **"HiKO and Neural Permutation Selection: A Hierarchical Framework for Beyond-Second-Order KO and Reed-Muller Codes"** by Shubham Srivastava and Adrish Banerjee (Department of Electrical Engineering, IIT Kanpur).

Kronecker Operation (KO) codes replace the Plotkin construction of Reed-Muller (RM) codes with neural networks and beat low-order RM codes, but extending them to higher orders (r > 2) is not straightforward. The paper makes two contributions:

1. **HiKO** – a hierarchical training framework that transfers pre-trained constituent KO codes (KO(4,3), KO(4,2)) into larger codes (m = 8, 9; r = 3, 4) through progressive unfreezing, with learnable skip scales and dropout. It gives modest BER gains over RM codes (about 0.05–0.15 dB at BER 10⁻⁴).
2. **Neural permutation-selected Dumer decoding** – an error analysis shows that the remaining errors of Dumer decoding of RM codes are decoder failures concentrated at the first decoded leaf, and that a different ordering of the received symbols often avoids them. A fixed check-based front-end (FE) plus a small MLP selects one coordinate permutation from the code's affine automorphism group before a single, unchanged Dumer pass. On RM(8,3), RM(8,4), RM(9,3) and RM(9,4) this gives 0.65–0.92 dB SNR gain over plain Dumer decoding at BLER 10⁻²/10⁻³ with one Dumer pass (0.95–1.35 dB with the top-2 variant, which needs two passes).

## Repository structure

| Path | Description |
|------|-------------|
| `HiKO-8_3.ipynb`, `HiKO-8_4.ipynb`, `HiKO-9_3.ipynb`, `HiKO-9_4.ipynb` | HiKO training and evaluation for the codes (m, r) = (8,3), (8,4), (9,3), (9,4). |
| `RMcode-8,3.ipynb`, `RMcode-8,4.ipynb`, `RMcode-9,3.ipynb`, `RMcode-9,4.ipynb` | RM code baselines with Dumer decoding for the same parameters. |
| `selector_notebooks/` | Neural permutation-selected decoding. |
| `selector_notebooks/HiKO_PermSelector-{8,3 / 8,4 / 9,3 / 9,4}.ipynb` | Front-end + neural selector training and evaluation for each code. |
| `selector_notebooks/RM_Dumer_Baseline.ipynb` | Plain Dumer decoding baseline used for comparison. |
| `selector_notebooks/ResultsAndWeights/` | Saved results and trained selector weights (one folder per run). |
| `reed_muller_modules/` | Helper modules: RM codebook generation, Hadamard transform (FHT), communication utilities, logging, shared functions. |
| `data/` | Precomputed matrices used by the notebooks. |
| `data_loader.py` | Data generator utilities. |

## Method overview

### HiKO (hierarchical training)
- Same Dumer-style decoding flow as KO codes: decode the `v` branch first, take hard decisions, then decode `u`.
- Encoder nodes matching a pre-trained constituent code reuse its parameters; all other nodes use `g(u, v) = (u, g̃(u, v) + α·(u ⊕ v))` with SELU activations, dropout and a learnable skip scale `α`.
- Training phases: (1) train constituent codes independently, (2) map their parameters to matching nodes and freeze them, (3) unfreeze the j-th component at epoch ⌊jT/(K+1)⌋ with a restarted one-cycle learning-rate schedule.
- Trained over AWGN with Adam (η_max = 2×10⁻⁴) for 300 epochs.

### Neural permutation-selected decoding
1. A parameter-free check-based front-end (min-weight dual codewords, tanh message passing) refines the LLRs.
2. P = 16 affine permutations (identity + 15 fixed random) are applied; first-leaf features (23 per candidate, from the FHT of the first-leaf LLRs) are computed without decoding below that leaf.
3. An MLP (2 × 128 SELU, 25,601 parameters) scores the candidates, trained with a listwise loss on informative frames.
4. One unchanged Dumer pass decodes the selected permutation and the result is mapped back.

## Results (from the paper)

SNR gain (dB) over plain Dumer decoding at BLER 10⁻² / 10⁻³:

| Decoder | RM(8,3) | RM(8,4) | RM(9,3) | RM(9,4) |
|---------|---------|---------|---------|---------|
| FE + Dumer | 0.42 / 0.50 | 0.50 / 0.61 | 0.20 / 0.26 | 0.28 / 0.38 |
| FE + hand rule | 0.81 / 0.84 | 0.70 / 0.76 | 0.77 / 0.77 | 0.61 / 0.70 |
| FE + NN selector | 0.86 / 0.92 | 0.73 / 0.79 | 0.81 / 0.83 | 0.65 / 0.73 |
| FE + NN top-2 | 1.25 / 1.28 | 0.95 / 1.01 | 1.28 / 1.35 | 0.97 / 1.08 |

## Getting started

Requirements: Python 3, PyTorch, NumPy, SciPy, Matplotlib, Jupyter (a GPU is recommended).

```bash
git clone https://github.com/shubhamsrivast4u/General-Rate-KO-Codes.git
cd General-Rate-KO-Codes
jupyter notebook
```

- To reproduce HiKO results, open the `HiKO-<m>_<r>.ipynb` notebook for the desired code.
- To reproduce permutation-selected decoding, open `selector_notebooks/HiKO_PermSelector-<m>,<r>.ipynb`; `RM_Dumer_Baseline.ipynb` gives the plain Dumer reference.

## Pre-trained weights

Weights of the HiKO models: [OneDrive link](https://iitk-my.sharepoint.com/:f:/g/personal/shubhsr_iitk_ac_in/Emn2AGpR-YRNhqYW2a1-J9QB5L2try8oajNkAWHxE02YOg?e=gfpMb4). Trained selector weights are in `selector_notebooks/ResultsAndWeights/`.

## License

See [LICENSE](LICENSE).
