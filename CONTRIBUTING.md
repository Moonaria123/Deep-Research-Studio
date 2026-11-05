# Contributing

Thank you for improving Deep Research Studio.

## Before you start

1. Search existing issues and open a focused issue for a material behaviour change.
2. Keep changes inside the Gen1 Web, API, deployment, or documentation boundary.
3. Never add provider keys, local configuration, reports, databases, root keys, recordings with private data, or generated runtime directories.
4. Keep the hosted Demo deterministic, key-free, network-independent, and non-persistent.

## Development checks

For frontend changes:

```bash
cd apps/web
npm ci
npm test
npm run lint
npm run build
npm run build:demo
```

For backend changes, use a Python 3.12 virtual environment:

```bash
python -m pip install -r docker/requirements-api.txt -r requirements-dev.txt
python -m pytest tests -q
```

Run the targeted module first while developing, then run the complete published suite before opening the pull request.

For deployment changes, use a unique Compose project, ports, data directory, and key directory. Do not reuse a live installation for testing.

## Pull requests

- Explain the user-visible change and its boundary.
- List the commands you actually ran and their results.
- Separate mocked, offline, and real-provider evidence.
- Update both README languages when public behaviour or installation changes.
- Confirm the diff contains no credentials, personal data, generated caches, or unrelated changes.

By contributing, you agree that your contribution is licensed under the [PolyForm Noncommercial License 1.0.0](LICENSE). Redistributed original or modified copies must retain the original-author attribution and every `Required Notice:` line from [NOTICE](NOTICE), along with the licence or its URL. Retain any applicable third-party notices. Separate commercial licensing of a contribution requires permission from its rights holder; submitting a contribution does not assign its copyright. See [licensing](docs/licensing.md).
