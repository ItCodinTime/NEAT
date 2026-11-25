# Experimental v3 status

`nce_spec_v3` is an opt-in research surface. It does not change the stable v2 default.

The NumPy reference engine implements global and per-parameter conflict granularity,
trust-region correction bounds, opponent ensembles, correction/preconditioner ordering,
and explicitly ordered deterministic reductions. Keras and PyTorch adapters operate one
parameter tensor at a time, so they reject global granularity. They also reject ordered
deterministic reductions, which are reference-engine-only.

No v3 option is promoted as a default or included in headline performance claims. Promotion
requires a committed protocol, a correction-off control, equal tuning budgets for every
optimizer, cross-adapter parity where the semantics are implementable, and results that
transfer across more than one workload.
