# Security notes

Use only trusted model, processor, tokenizer, and checkpoint files. Downloading a model does not by itself establish its safety. Do not enable remote custom code or custom generation implementations from untrusted sources. Keep credentials out of model directories, logs, issues, and source control.

## Historical Transformers dependency

The original repository and model configuration used Transformers 5.3.0. During repository preparation, GitHub reported three high-severity advisories affecting that version:

| Advisory | Published affected range | Published patched version |
| :--- | :--- | :--- |
| [LightGlue model initialization code execution](https://github.com/advisories/GHSA-fgcw-684q-jj6r) | `<5.5.0` | 5.5.0 |
| [Chat-template path traversal in save_pretrained](https://github.com/advisories/GHSA-xrqw-3rrv-vx5w) | `<5.10.0` | 5.10.0 |
| [Custom generation code downloaded before trust consent](https://github.com/advisories/GHSA-x9r9-c232-4q39) | `>=4.49.0, <=5.8.1` | None explicitly listed |

New installations pin **Transformers 5.10.4**, which is outside the published affected ranges above; the yanked 5.10.0 release is intentionally avoided. The third advisory does not explicitly name a patched version, so this is not a claim that every loading path has been independently audited. Compatible mistral-common is pinned to 1.11.4. Check upstream advisories for later changes and additional issues.

Historical metadata remains unchanged for provenance. Do not treat it as a recommendation to install the old version. Read the [reproducibility notes](docs/reproducibility.md) before comparing a new environment with the paper experiments.

## Development tools

The development formatter is pinned to Black 26.10.1, following GitHub's upgrade guidance for the cache-filename and GitHub Action required-version advisories. This repository invokes the installed formatter directly, not the Black GitHub Action; keep the development pin updated as well as runtime dependencies.

## Reporting

For a suspected vulnerability, contact **cuibenlei.cbl@alibaba-inc.com**. Do not post credentials, private training data, or working exploit payloads in a public issue. This contact route does not imply a guaranteed response time or a bug-bounty program.
