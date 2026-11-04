# Contributing

Keep changes focused and preserve whole-family objective semantics. Discuss substantial training or data-contract changes before implementation.

## Development checks

After installing `requirements.txt`, install the formatter:

```bash
python -m pip install -r requirements-dev.txt
python -m black --check train_dreamrealm.py dreamrealm scripts tests
python -m unittest discover -s tests -v
python scripts/local_smoke.py
python -m compileall -q train_dreamrealm.py dreamrealm scripts tests
```

Use `python -m black train_dreamrealm.py dreamrealm scripts tests` to format changes. GPU experiments are separate from CPU correctness checks; state exactly which ones were run.

For bugs, provide versions, relevant arguments, expected/actual behavior, and a minimal synthetic reproduction. Remove credentials, internal paths/endpoints, private images, and personal data from logs. Do not upload checkpoints or training datasets in pull requests.

Contributions are made under the repository's Apache-2.0 license. Third-party code must retain applicable notices and license terms.
