# GitHub repository settings

Apply `.github/rulesets/main.json` after the commit containing the `CI required` and `Docs`
checks reaches the default branch:

```bash
gh auth login
gh api --method POST repos/ItCodinTime/NEAT/rulesets \
  --input .github/rulesets/main.json
```

Then verify the ruleset is active:

```bash
gh api repos/ItCodinTime/NEAT/rulesets
```

Under **Settings → Code security and analysis**, enable Dependabot alerts, secret scanning,
and push protection. Under **Settings → Actions → General**, keep the default workflow token
read-only; `release.yml` requests write or OIDC permissions only for the jobs that need them.

Add new collaborators with Triage or Write access. Do not grant Maintain or Admin solely to
allow merging; the ruleset and CODEOWNERS review handle normal contributions.
