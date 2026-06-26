# Algorithm

Standard NEAT is built around a single idea: if the current gradient is moving
against an opponent proxy, add a bounded correction term before momentum and
parameter updates.

Design choices for the first release:

- projection-based correction is the default
- cosine-only correction is retained as a simpler ablation mode
- the correction term is clipped relative to the gradient norm
- decoupled weight decay is the default behavior
- the correction can be delayed with `correction_warmup_steps`
- the correction can be gated with `conflict_threshold`
- `adaptive_alpha=True` makes the correction strength respond to conflict,
  gradient-noise, and alignment EMAs
- `adaptive_preconditioning=True` adds a diagonal second-moment preconditioner
  for the update and, optionally, the correction itself
- `update_mode="lion"` switches the final update to a Lion-style sign step
- `gradient_centralization=True`, `nesterov=True`, and `lookahead_k > 0` enable
  standard optimizer stabilizers as controlled ablations

Opponent proxy options for the standard optimizer:

- `momentum`: previous momentum vector
- `previous_gradient`: previous raw gradient
- `gradient_ema`: exponential moving average of recent gradients
- `blended`: convex blend of momentum and gradient EMA

The adaptive alpha path is deliberately tensor-local. Each parameter tensor
tracks its own conflict, gradient-noise, and alignment signals, so early and
late layers can receive different effective correction strengths without
adding a new layer registry.

The diagonal preconditioner is a cheap Fisher/Hessian-diagonal approximation
based on the gradient second moment. It gives NEAT a second-order element while
staying compatible with Keras optimizer state and the NumPy reference engine.
It should not be described as full Newton, exact Hessian, or exact Sophia.

The benchmark diagnostics now report mean conflict ratio, mean correction
ratio, mean update alignment, mean opponent norm, and the fraction of steps
where the correction was active. On the current digits benchmark, those
numbers show that the correction is small and rarely changes the update
direction materially, which explains why NEAT is still trailing Adam and SGD
there.

The repository also includes a separate player-aware mode for explicit
per-example or per-task gradients. That path treats each player as a distinct
gradient contributor and forms a leave-one-out opponent proxy from the
remaining players before aggregation.

## Not Yet Claimed

The current repo does not claim first-class schedule-free training, automatic
LoRA injection, or production-scale ImageNet/LLM/RL wins. The code now has the
optimizer hooks needed to benchmark those directions, but credible claims
require reproducible runs and published logs.
