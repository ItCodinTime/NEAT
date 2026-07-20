# Regeneron STS experiment journal

This journal records operational decisions and exclusions. Raw run manifests
remain the authoritative source for numerical results.

## 2026-07-18

### Repository audit

- Restored the saved GitHub Desktop changes and resolved conflicts in the
  changelog, README, documentation index, and quickstart.
- Verified 75 full tests passed with 6 optional-dependency skips before STS
  development. The existing formatting and strict-type-checking debt was
  recorded separately and not mixed into the experiment change.
- Created branch `codex/regeneron-sts`. Merge restoration is commit `b580d35`;
  the initial STS harness and protocol are commit `5dd77b5`.

### Vultr provisioning

- Confirmed an empty account with USD 200 credit and a project hard cap of
  USD 100. No inference endpoint was required for model training.
- Attempted an available 8 GB fractional A16 GPU. Vultr rejected the request
  because Cloud GPU access requires a support request. No support request was
  submitted and no GPU service was created.
- Provisioned `vhf-8c-32gb` in New Jersey at USD 0.263/hour: 8 vCPUs, 32 GB RAM,
  Ubuntu 24.04, Python 3.12. Access used a dedicated ED25519 key and firewall
  group restricted to two explicit `/32` management sources.
- Attempted to replace the serial worker with four smaller identical workers at
  the same aggregate hourly price. Vultr accepted the create requests but locked
  three services under the account's monthly allocation limit. The locked
  services were deleted immediately. The runnable small worker could not be
  upgraded under the same quota, so it was deleted and the original 8-core plan
  was reprovisioned after the quota released.
- Conclusion: use one serial 8-core worker. Do not request a quota increase or
  copy the account API key to the instance. The USD 80 stop-and-review threshold
  remains in force.

### Engineering and smoke evidence

- Added deterministic paired-image datasets, a shared two-head CNN, Adam,
  PCGrad, exact two-task MGDA, correction-only NEAT, validation-only tuning,
  paired statistical reporting, mechanism plots, resource measurement, and
  environment capture.
- Vultr lint passed and 83 unit tests passed after the full harness was added.
- Ran an eight-cell preliminary smoke matrix: two datasets by four methods, one
  seed, one epoch, 512 training pairs, and 256 validation/test pairs. All cells
  wrote complete manifests. They are excluded from paper aggregation because
  they are explicitly marked preliminary and were created before git metadata
  was present on the worker.
- A separate Multi-MNIST learning check using Adam and 5,000 training pairs
  improved validation average accuracy from `0.38325` after epoch 1 to `0.70000`
  after epoch 3. This confirmed the generated task and model can learn before
  tuning began. The check is preliminary and excluded from paper claims.

### Next decision gate

Bootstrap the final serial worker from a clean committed revision, verify git
and dataset provenance, then run the declared validation-only learning-rate
grid. Tier 1 remains blocked until `locked_lrs.json` is complete.

## 2026-07-20

### Completed v1 result and v2 diagnosis

- The full tuning, Tier 1, mechanism, and ablation matrices completed and were
  synchronized locally before the Vultr worker was deleted.
- Base correction-only NEAT substantially underperformed all three baselines.
  Correction-off NEAT behaved almost identically, while sign transport nearly
  recovered baseline performance. The mechanism correlation was negative.
- Code review established that v1 NEAT received an already averaged task
  gradient and measured conflict against historical momentum. It did not see
  the per-task gradients used by PCGrad and MGDA. Its learning-rate grid also
  ended at `0.003`, with the selected value on that boundary.

### Exploratory task-aware v2

- Added a global shared-trunk soft projection over explicit per-task gradients.
  The projection uses linear conflict scaling and feeds an unchanged Adam or
  published Lion transport.
- Added exact `alpha=0` transport controls, per-step task-conflict diagnostics,
  a validation-only promotion matrix, and mathematical unit tests.
- V2 is a post-result method revision. It is not pooled with the preregistered
  v1 study and cannot retroactively confirm the original hypothesis.
