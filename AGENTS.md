# VehicleAI repository rules

These rules apply to every human or Codex-assisted change in this repository.

## Before development

1. Run `git status --short --branch`, inspect the current branch, and fetch the remote before starting substantial work.
2. Never develop new features directly on `main`. Start from `develop` and use `feature/<short-name>`; use `fix/<short-name>` for corrections.
3. Preserve `main` as the stable branch. Never force-push, rewrite shared history, or delete a remote branch without explicit user approval.
4. Stop and report authentication failures, remote divergence, unexpected sensitive files, history-rewrite requirements, or any risk of data loss.

## Changes and tests

1. Keep commits focused and give them clear imperative or conventional commit messages.
2. After completing development, run the relevant focused tests and the complete portable suite with `python -m pytest -m "not cuda"`. Run CUDA-marked tests locally when the change affects GPU code and compatible hardware is available.
3. When all required tests pass, update `CHANGELOG.md`, create a clear commit, and push the current feature/fix branch to `origin` without waiting for a routine reminder. Stop instead if authentication fails, the remote has diverged, or the push would overwrite remote work.
4. Do not create a commit or formal release when required tests fail; report the failure.
5. For a formal version, update `README.md`, `CHANGELOG.md`, and the version in `pyproject.toml` together.
6. Use semantic versions such as `v0.3.1` and `v0.4.0`. Small daily changes may be committed and pushed without changing the version.

## Data and model safety

1. Never commit raw images or videos, candidate crops, prepared datasets, human `annotations.jsonl`, curated manifests containing local records, embeddings, predictions, results, virtual environments, caches, credentials, or large model weights.
2. Review `git status --ignored` and staged files before every commit. If sensitive content has entered history, stop and report it before attempting any history rewrite.
3. Never delete local project data as part of Git cleanup.
4. AI predictions must remain separate from human annotations and must never overwrite a human label.

## Publishing

1. Push feature/fix branches and open a reviewed pull request into `develop`.
2. After integration tests and version acceptance, merge `develop` into `main` through a reviewed pull request.
3. Create an annotated Git tag only for an exact, tested release commit and push it normally without force.
4. Create a GitHub Release only after the tag tests pass and the user explicitly confirms release creation. Release notes must include changes, test results, and known limitations.
5. GitHub Actions must not auto-merge feature branches or bypass protected-environment approval.

See `docs/RELEASING.md` for the complete branch and release procedure.
