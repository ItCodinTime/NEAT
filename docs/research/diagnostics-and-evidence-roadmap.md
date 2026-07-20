# Diagnostics and evidence roadmap

## Decision basis

This roadmap was written after the completed v1 five-seed study and the
validation-only v2 promotion gate. V1 collapsed primarily because its transport
was mismatched, while task-aware v2 produced correction ratios from roughly
`0.02` to `0.075` without a transferable accuracy gain. Those results rule out
shipping stronger correction or new presets merely because they sound useful.

## Add now

1. **Keras History diagnostics and portable reports.** The optimizer already
   exposed aggregate diagnostics, but they did not appear in the object returned
   by `model.fit`. `NEATDiagnosticsCallback` now adds namespaced epoch metrics,
   records the full epoch log, and can write JSON or self-contained HTML.
2. **Gradient, correction, and step norms.** Conflict ratios alone cannot
   distinguish an inactive correction from a transport-scale problem. Keras and
   PyTorch adapters now report gradient norm, correction norm, step norm, and
   effective step size.
3. **Controlled conflict-severity evidence.** The validation-only severity
   phase changes image overlap, directly measures shared-trunk task-gradient
   cosine, and compares task-aware alpha `0.5` against alpha `0` under identical
   Adam transport. It does not use the test split.
4. **Cross-backend smoke validation.** Diagnostics should be checked under the
   Keras TensorFlow, JAX, and PyTorch backends before claiming Keras 3
   interoperability.

## Already present

- Conflict gating through `conflict_threshold`.
- Adaptive strength through `adaptive_alpha` and `adaptive_correction`.
- TensorBoard logging through `NEATDiagnosticsCallback`.
- Reproducible vision, language, diffusion, reinforcement-learning, synthetic,
  STS, profiling, and reporting harnesses. Coverage does not imply every
  workload has a publishable completed result.
- Matched correction-off controls, timing, memory, paired tests, and mechanism
  diagnostics in the STS suite.

## Defer

- **Presets and new defaults.** No corrected configuration has passed the
  cross-dataset promotion rule. A named `stable` or `balanced` preset would
  imply evidence the project does not have.
- **Conflict early stopping or scheduling.** Diagnostics should first show that
  an intervention based on those signals predicts validation improvement.
- **Nightly GPU farms and public dashboards.** Useful after a smaller suite is
  stable, but premature while cloud GPU access is unavailable and the main
  formulation remains a negative result.
- **Large-model, diffusion, and broad real-world sweeps.** These are expensive
  ways to search for a positive example. A controlled regime map should locate
  a plausible benefit region first.

## Do not implement as described

- **Automatic per-task detection inside `NEAT.update_step`.** Keras passes an
  optimizer the gradient of the already-aggregated compiled loss. Task identity
  is no longer available there. Honest native multi-task support requires a
  trainer or custom `Model.train_step` that computes each loss gradient before
  aggregation.
- **Automatic fallback to Adam as a hidden behavior.** Conflict gating already
  skips correction, but default NEAT uses momentum transport rather than Adam.
  Silently switching optimizer families would change state and semantics. A
  future task-aware wrapper may use an explicit Adam transport control, as v2
  already does.

## Cloud gate and budget

- Account pending charges at the start of this phase: USD `14.06`.
- Project ceiling: USD `100`; remaining headroom: approximately USD `85.94`.
- Stage A: one short CPU worker for three-backend smoke checks and a reduced
  severity matrix.
- Stage B: expand to three seeds and both datasets only if Stage A shows that
  overlap changes measured task conflict and all artifacts are complete.
- Stop and delete workers if the intervention fails to change measured conflict,
  any alpha-zero control diverges, provenance is incomplete, or projected total
  charges exceed USD `90`.

## Claim boundary

Diagnostics explain what the optimizer did. They do not establish that the
correction caused a performance change. That claim requires a matched transport
control, paired seeds, and a relationship between the manipulated conflict,
measured correction, and held-out validation gain.
