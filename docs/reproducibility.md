# Reproducibility and release scope

## What this repository provides

This release contains the **RealmMatch local training implementation**: whole-family validation, full-answer teacher forcing, matched-pair SFT, bidirectional row/column contrastive objectives, score replay, CPU-offloaded AdamW, and checksummed checkpoint I/O. It does not contain RealmForgePipe's end-to-end mining/editing/human-review system, the training images or annotations, a complete benchmark harness, or the distributed launchers used for the paper experiments.

The released model is available independently on [Hugging Face](https://huggingface.co/Alibaba-VELLDEPTH/Qwen35-9b-DreamRealm). Do not infer that the included single-GPU example command exactly reconstructs that checkpoint or all reported results.

## Objective invariants

For an edited family, rows in the implementation index images and columns index answers. Every image is scored against every answer, with the correct correspondences on the diagonal. Both row-wise and column-wise normalization use the complete family. The no-edit branch retains the correct answer and available original-image negative answers, with no column-selection term.

Two-pass replay computes first-order gradients of the same whole-family objective without retaining every cell's activation graph at once. This equivalence depends on unchanged parameters and deterministic scores between passes. It is not a higher-order differentiation method. The trainer disables dropout and rejects replay drift beyond its numerical tolerance.

Losses are averaged within a family and then across families in the optimizer step. Do not normalize by the number of matrix cells, split families into independent softmax chunks, or change the answer/EOS masking when comparing objectives.

## What the tests establish

The CPU suite checks full-gradient agreement against joint autograd over variable-sized families; row/column signals and permutation invariance; no-edit boundaries; optimizer resume equivalence; family rejection; stochastic-replay detection; vision freezing; and checkpoint round-trip/corruption rejection. The smoke test exercises a tiny synthetic linear scorer, not Qwen.

These tests establish implementation properties, **not** model quality, dataset correctness, real-GPU memory fit, training throughput, or end-to-end reproducibility of the paper experiments. CI also runs these CPU checks only.

## Record for a comparable experiment

### Dependency update for new installations

The model release configuration records Transformers 5.3.0. The repository now pins Transformers 5.10.4 and mistral-common 1.11.4 for new installations, following the historical Transformers version's security alerts. Version 5.10.0 was not selected because PyPI marks that release as yanked. Upgrading dependencies can affect numerical behavior; this update must not be represented as an exact reconstruction of the paper's historical software environment. Re-run real-model GPU admission and benchmark evaluation before claiming numerical equivalence. See [security notes](../SECURITY.md).

Local validation of the updated dependencies covered model-class imports, loading the released configuration and processor, non-thinking answer/EOS encoding on CPU, all nine unit tests, and the smoke test. No full 9B weight forward/backward pass or benchmark rerun was performed as part of this repository preparation.

### Experiment records

Retain the exact model revision, processor/chat template, data manifest and image hashes, split membership, construction filters, family membership, sampling seed, optimizer/scheduler settings, frozen parameters, precision, framework/CUDA versions, hardware, and evaluation prompts/scorers. Prevent train/test image and near-duplicate leakage. Report deviations from the manuscript's protocol.

The local trainer records the JSONL SHA256, arguments, library versions, GPU name, family statistics, step metrics, and parameter probes. It does **not** hash every referenced image or independently audit dataset leakage. Record those separately in your own experiment system.

## Responsible use

Improved benchmark scores do not guarantee that hallucinations are eliminated. Validate on your intended distribution and retain human review for consequential uses. Only use images and annotations you have rights to process and distribute. Never commit private datasets, access tokens, internal endpoints, or model weight files to this code repository.
