# v3 specification status

The stable default remains `nce_spec_v2`. No v2 equation, default, or stored
worked example is replaced by the v3 work.

The v3 options below are selectable only with `spec_version="nce_spec_v3"`.
Every option is default-off; attempting to enable one under v2 is a
configuration error. See [v3 author decisions](v3_decisions.md).

## Conflict granularity

`conflict_granularity="per_parameter"` marks conflict calculations as local to
the parameter tensor or parameter group supplied to an engine step. The
default `"global"` retains the existing whole-input reduction. This controls
the scope of the dot products and norms; it does not change the projection
formula.

## Trust-region correction bound

`trust_region_ratio > 0` replaces `nce_clip_ratio` as the active L2 bound. The
correction is rescaled when its norm exceeds `trust_region_ratio * ||g||`.
Setting it to zero retains the v2 clipping control.

## Opponent ensemble

`opponent_ensemble=True` combines momentum and gradient-EMA state. The initial
momentum weight is `opponent_ensemble_momentum_weight`; when schedule steps are
positive, that weight moves linearly to one over the specified number of
steps. No learned parameters are introduced.

## Correction-aware preconditioning order

`correction_preconditioning_order="correction_then_precondition"` computes the
projection in raw gradient coordinates and leaves diagonal preconditioning to
the final update transport. `"legacy"` computes and maps the correction using
the existing preconditioned coordinates.

## Deterministic reference mode

`deterministic=True` makes reference-engine dot products and norms use explicit
ordered float32 reductions. It changes reduction mechanics only and does not
enable global deterministic settings in framework runtimes.
