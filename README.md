# repo-digest-workflows

A [Mistral Workflows](https://docs.mistral.ai/studio/workflows/overview) project that builds a
catch-up digest for a GitHub repo and posts it to your Slack self-DM.

## What the workflow does

Workflow `repo-digest` takes two inputs:

```json
{
  "repo": "owner/repo",
  "since": "2026-01-01"
}
```

It then:

1. Searches GitHub for pull requests `is:merged merged:>=<since>` and for pull requests
   `is:open -is:draft` (waiting on review) on the repo.
2. Writes a one-line summary for each PR with a Mistral model (falls back to the PR title).
3. Posts the digest to the triggering user's Slack self-DM (the conversation with yourself),
   one bullet per PR with a one-line summary and a link.

The workflow is **on-behalf-of**: it resolves the triggering user's credentials for the
`github_app` and `slack` connectors, so it must run on a **hardened** managed deployment.
The Slack connector is OAuth-only, so the triggering user connects Slack once (Studio →
Connectors, or via the auth URL the first execution emits).

## Layout

```text
apps/worker/                 Python worker (mistralai-workflows SDK)
  src/worker/workflows/      the repo-digest workflow
  src/worker/entrypoints/    worker entrypoint (discovers worker.workflows)
  tests/                     pytest suite (Temporal time-skipping test env)
deploy/docker/Dockerfile.worker   image built by the managed deployment
```

## Develop

```bash
cd apps/worker
uv sync
uv run pytest
```

Local worker (needs `MISTRAL_API_KEY` in `.env`, see `.env.example`):

```bash
cd apps/worker
uv run --env-file ../../.env python -m worker.entrypoints.worker
```

## Deploy (managed, Mistral Cloud)

Push this repository to GitHub, then create a managed deployment with build context `.` and
Dockerfile `deploy/docker/Dockerfile.worker`, `hardened=true`, pinned to a pushed commit SHA.
The Mistral GitHub App must be installed on the repository so Mistral Cloud can clone it.

## Run

Start an execution of `repo-digest` (Studio console or API) with the input above. The first
run as a user without a Slack credential pauses and emits an `auth_url` event — complete the
OAuth flow and the run resumes.

## Test 1
