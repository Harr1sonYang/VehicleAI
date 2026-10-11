# VehicleAI Git and release workflow

## Branches

- `main`: stable, accepted releases only.
- `develop`: integration branch for completed feature and fix work.
- `feature/<short-name>`: one new feature, branched from `develop`.
- `fix/<short-name>`: one correction, branched from `develop`.
- `v0.2-classification`: retained as the historical branch that produced the first complete post-v0.1 snapshot. Do not delete or rewrite it.

Use pull requests for `feature/*` or `fix/*` into `develop`, and for an accepted release from `develop` into `main`. No workflow in this repository automatically merges a pull request.

## Daily change

1. Fetch the remote and inspect `git status`, the current branch, and upstream divergence.
2. Create a feature or fix branch from current `develop`.
3. Make a focused change and update tests/documentation.
4. Run `python -m pytest -m "not cuda"`; run `python -m pytest -m cuda` on a configured CUDA machine when relevant.
5. Inspect staged paths and `git status --ignored` for data, secrets, weights, or personal files.
6. Commit and push the branch, then open a pull request into `develop`.

## Formal release

1. Choose a semantic version and update `pyproject.toml`, `README.md`, and `CHANGELOG.md`.
2. Confirm all pull-request checks pass and run appropriate local CUDA validation.
3. Merge `develop` into `main` only after review and version acceptance.
4. On the exact release commit, create an annotated tag: `git tag -a vX.Y.Z -m "VehicleAI vX.Y.Z"`.
5. Push `main` and the tag without force.
6. Manually run the `Create release` workflow for that existing tag. Its tests must pass before the GitHub Release job can run.
7. Include changes, test counts, hardware-specific validation, and known limitations in release notes.

Never manufacture a historical tag from a newer snapshot. If an exact commit or trustworthy snapshot does not exist, document the milestone but do not create a tag or release for it.

## Recovery and safety

Never use force push, destructive reset, history filtering, or credential removal from history without explicit approval. If a secret or private dataset is discovered in a commit, stop publishing, report the affected commit/path, and agree on a recovery plan first.
