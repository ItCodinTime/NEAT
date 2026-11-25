# Governance

NEAT uses maintainer-led governance. The repository owner is the final decision-maker
for releases, security responses, public API changes, and changes to the versioned
mathematical specification.

## Access levels

- **Triage** may label and organize issues without changing code.
- **Write** may create branches and pull requests. Direct pushes to `main` are not allowed.
- **Maintain** may manage issues and pull requests but may not bypass required reviews.
- **Admin** is reserved for the repository owner and a designated recovery maintainer.

New collaborators should start with Triage or Write access. Access is reviewed when a
collaborator becomes inactive or changes responsibilities.

## Decisions and review

All code changes use pull requests. At least one approving review, passing required checks,
resolved review threads, and CODEOWNERS approval are required before merge. Authors may not
approve their own last push. Squash merge is preferred for a readable history.

Changes to the update rule must update the reference implementation, math specification,
tests, and compatibility notes together. Security fixes follow `SECURITY.md` and may use a
private fork until coordinated disclosure.

## Releases

Releases follow Semantic Versioning and use `vMAJOR.MINOR.PATCH` tags. The release checklist
in `docs/contributing/release.md` is authoritative.
