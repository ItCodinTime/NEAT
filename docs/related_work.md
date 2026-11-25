# Related work

NEAT is a historical-gradient correction method: its standard optimizer compares the
current gradient with state accumulated along the same optimization trajectory. This is
different from multi-task gradient-surgery methods, which compare gradients from separate
objectives at the same training step.

## PCGrad and CAGrad

[PCGrad](https://arxiv.org/abs/2001.06782) projects one task gradient away from another when
their dot product is negative. [CAGrad](https://arxiv.org/abs/2110.14048) chooses an update
that balances progress on the average objective against the worst local improvement among
tasks and provides convergence analysis for that objective.

NEAT's standard `model.compile(...)` path does not have access to separate task gradients.
It projects against momentum, the previous gradient, or a gradient EMA instead. The optional
player-aware API can consume separate gradients, but it remains a bounded soft-projection
heuristic rather than an implementation of PCGrad or CAGrad.

## Gradient Vaccine

[Gradient Vaccine](https://arxiv.org/abs/2010.05874) uses measured task relationships to
encourage gradient alignment in multilingual multi-task learning. It relies on simultaneous
task gradients and a target similarity. NEAT instead reacts only to negative alignment and,
in its standard mode, uses optimization history rather than task identity.

## Nash-MTL

[Nash-MTL](https://proceedings.mlr.press/v162/navon22a.html) formulates multi-task gradient
combination as a bargaining game and derives an update from the Nash bargaining solution.
NEAT does not solve that bargaining problem and does not claim Nash-MTL's equilibrium or
convergence guarantees. “Nash-inspired” in NEAT refers to treating directional disagreement
as a two-signal interaction; it is an interpretation of the correction, not an exact game-
theoretic solver.

## Composed optimizer techniques

NEAT optionally composes its correction with established mechanisms: Lion-style sign
updates, [Lookahead](https://arxiv.org/abs/1907.08610), gradient centralization, Nesterov
momentum, and Adam-style second-moment preconditioning. These are exposed as controlled
options and ablations, not presented as new contributions.

## Claim boundary

The distinct contribution evaluated by this repository is the bounded correction against a
historical opponent signal, together with diagnostics that measure when that correction is
active. Performance claims require matched correction-off controls and equally tuned
baselines; the presence of a correction alone is not evidence of improved generalization.
