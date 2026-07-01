# CIFAR-10 ResNet-18 MPS Study — 2026-06-29

This is a bounded local comparison of NEAT and AdamW. It is useful evidence,
but it is not a full convergence study and does not support a state-of-the-art
claim.

## Protocol

- Dataset: CIFAR-10, full 50,000-example training set and 10,000-example test set
- Model: `timm` ResNet-18, trained from scratch
- Hardware: Apple MPS on macOS 26.5.1 ARM64
- Runtime: Python 3.13.6, PyTorch 2.12.0
- Batch size: 256
- Seeds: 7, 11, and 19
- Schedule: three epochs with cosine annealing
- Learning-rate tuning grid: `3e-4` and `1e-3`, evaluated on seed 7
- Selected learning rate: `1e-3` for both optimizers
- Weight decay: `5e-4`
- Target accuracy: 70%; neither optimizer reached it in three epochs

The model, augmentations, data order, batch size, precision, schedule, and
learning-rate search grid were held constant between optimizers.

## Final Results

| Optimizer | Top-1 accuracy | Final train loss | Final loss variance | Run time |
|---|---:|---:|---:|---:|
| AdamW | 66.11% ± 0.43% | 0.9826 ± 0.0088 | 0.00464 ± 0.00026 | 383 ± 75 s |
| NEAT | 66.45% ± 0.17% | 0.9876 ± 0.0110 | 0.00496 ± 0.00053 | 831 ± 48 s |

Seed-level final Top-1 accuracy:

| Seed | AdamW | NEAT | NEAT − AdamW |
|---:|---:|---:|---:|
| 7 | 66.38% | 66.42% | +0.04 pp |
| 11 | 66.34% | 66.29% | −0.05 pp |
| 19 | 65.61% | 66.63% | +1.02 pp |

The paired mean accuracy difference was +0.34 percentage points for NEAT.
With only three seeds it was not statistically significant (paired t-test,
`t(2)=0.983`, `p=0.429`).

Mean accuracy by epoch:

| Epoch | AdamW | NEAT |
|---:|---:|---:|
| 1 | 53.96% | 53.87% |
| 2 | 62.27% | 62.02% |
| 3 | 66.11% | 66.45% |

NEAT did not converge faster during this run. It was slightly behind through
epoch 2 and slightly ahead at epoch 3. NEAT took 2.17 times as long as AdamW on
MPS and had 6.8% higher final minibatch-loss variance.

## NEAT Diagnostics

Across the three runs, the correction was active for 21.3% of parameter
updates. Its mean norm was only 0.133% of the gradient norm. The small
correction explains why the optimization curves are close to AdamW.

## Conclusion

This study finds no credible performance advantage over AdamW. Final accuracy
was nominally higher for NEAT, but the difference is within seed variation,
training loss and loss variance were slightly worse, no convergence-speed
advantage appeared, and MPS runtime was substantially worse.

The next credible step is a 100–200 epoch CUDA run with a wider independently
tuned learning-rate grid and at least three seeds. Until that exists, the
correct claim is parity on this short CIFAR-10 study, not superiority.

Raw manifests and curves from the local run are under
`benchmark-runs/multiepoch` and `benchmark-runs/multiepoch-report`.
