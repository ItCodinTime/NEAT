# Security Policy

## Supported Versions

| Version | Supported |
|---|---|
| `main` (development) | Yes |
| Latest tagged release | Yes |
| Earlier releases | No |

Security fixes are applied to `main` and the most recent tagged release. Older releases do not receive backports.

## Reporting a Vulnerability

**Do not open a public GitHub issue for security vulnerabilities.**

Report suspected vulnerabilities privately via [GitHub Security Advisories](https://github.com/ItCodinTime/NEAT/security/advisories/new) for this repository.

Please include:

- The `neat_optim.__version__` and Keras/backend version affected
- A clear description of the vulnerability and its potential impact
- Reproduction steps or a minimal proof of concept
- Any suggested mitigation or fix, if you have one

Maintainers will acknowledge receipt within 5 business days, validate the report, and coordinate a fix and public disclosure timeline. We aim to release a fix within 90 days of a confirmed vulnerability.

## Scope

NEAT is a numerical optimization library with no network access, no authentication, and no persistent storage. The most likely security-relevant issues are:

- Unsafe deserialization when loading Keras model checkpoints containing NEAT optimizer state
- Numerical instability that could be triggered by adversarially crafted inputs in a deployed model

If you are unsure whether an issue qualifies, err on the side of reporting it privately.
