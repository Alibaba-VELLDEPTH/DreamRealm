# Training guide

## Scope and requirements

`train_dreamrealm.py` is a single-GPU implementation of the RealmMatch family objective. It is not a distributed launcher and does not construct edited images. Prepare your own complete families using the [data format](data-format.md).

Real training requires Linux, a CUDA-capable GPU, locally downloaded Qwen3.5-9B model/processor files, and sufficient CPU RAM for FP32 master weights and AdamW states. Install `requirements.txt` after selecting the appropriate CUDA PyTorch build. A successful CPU smoke test does not establish that a real-model family fits in GPU memory.

## Preflight the actual images and tokenizer

```bash
python train_dreamrealm.py \
  --model /models/Qwen3.5-9B \
  --data /datasets/dreamrealm/families.jsonl \
  --media-root /datasets/dreamrealm \
  --output /checkpoints/dreamrealm-preflight-001 \
  --preflight-only
```

Every `--output` directory must be **new and nonexistent**, including preflight and resume runs. Preflight checks image access, chat-template boundaries, full-answer token masks, family shapes, and sequence lengths; it writes `preflight.json`. It does **not** load the model or run a forward/backward pass, so it is not a memory or throughput test.

## Run a new experiment

```bash
CUDA_VISIBLE_DEVICES=0 python train_dreamrealm.py \
  --model /models/Qwen3.5-9B \
  --data /datasets/dreamrealm/families.jsonl \
  --media-root /datasets/dreamrealm \
  --output /checkpoints/dreamrealm-run-001 \
  --epochs 3 --families-per-step 1 \
  --lr 5e-6 --lr-scheduler cosine --warmup-ratio 0.03 \
  --tau 1.0 --row-weight 0.5 --col-weight 0.5 \
  --max-length 6144 --max-pixels 262144 \
  --save-optimizer
```

Do not use `torchrun`: `WORLD_SIZE != 1` is rejected. Run a short real-model experiment on representative family sizes before a long run. `--max-steps` changes the scheduler horizon as well as the stopping point; a short admission test is a separate experiment, not an exact prefix of a differently scheduled long run.

### Objective and execution

- **Edited families:** an entire `N × N` image/answer score matrix, diagonal full-answer SFT, row-wise answer selection, and column-wise supporting-image selection.
- **No-edit families:** a `1 × (M+1)` matrix, correct-answer SFT and row contrast against available negatives. The column term is zero.
- Scores are mean log-likelihoods of the supplied answer plus EOS, with prompt/image tokens excluded. Overlength sequences are rejected, never silently truncated.
- Two-pass replay first computes all detached scores, then recomputes each cell with its full-family gradient coefficient. No per-cell softmax approximation is used. Dropout is disabled and score drift is checked.
- `--families-per-step` averages complete family losses, not individual matrix cells. Parameters update only after all selected families are accumulated.
- Vision patch embedding and transformer blocks are frozen; the visual merger and language parameters remain trainable. AdamW FP32 master weights and optimizer states live on CPU.

### Selected defaults

| Flag | Default | Meaning |
| :--- | ---: | :--- |
| `--epochs` | 1 | Epoch count; examples above explicitly use 3 |
| `--families-per-step` | 1 | Whole families per optimizer update |
| `--lr` | 5e-6 | AdamW learning rate |
| `--lr-scheduler` | constant | Examples explicitly select cosine |
| `--warmup-ratio` | 0 | Fraction of total steps used for warmup |
| `--tau` | 1.0 | Contrastive temperature |
| `--row-weight`, `--col-weight` | 0.5 each | Contrastive loss weights |
| `--max-length` | 6144 | Hard sequence-length limit |
| `--max-pixels` | 262144 | Processor image-pixel setting |
| `--max-grad-norm` | 1.0 | Gradient clipping bound |
| `--gpu-memory-fraction` | 0.72 | PyTorch process allocation fraction, not a fit guarantee |

Use `python train_dreamrealm.py --help` for the complete interface.

## Outputs

```text
dreamrealm-run-001/
├── run_config.json       # Arguments, input SHA256, versions, GPU, objective
├── steps.jsonl           # Family losses, gradient norms, timing, memory probes
├── summary.json          # Completion summary
├── parameter_probes.pt   # Initial/final sampled parameter values
├── model/               # Model + processor (unless --no-save-model)
└── optimizer_parts/     # Checksummed optimizer state (with --save-optimizer)
```

Checkpoint files are saved at the end of the run, not periodically. An interrupted process may have logs without a resumable checkpoint. Checksums protect checkpoint parts against incomplete/corrupted writes; only load checkpoints from sources you trust.

## Resume

Use a checkpoint produced with **both model and optimizer saving enabled**. Keep the data file unchanged and retain the same objective, seed, ordering, family accumulation, image/token limits, optimizer settings, clipping, checkpointing mode, and scheduler horizon. The trainer refuses contract changes.

Repeat the original command with a **new output directory**, adding:

```bash
--resume /checkpoints/dreamrealm-run-001 \
--output /checkpoints/dreamrealm-run-001-resumed
```

Do not append a second `--output` to a copied command; replace the original value. `--model` must still point to the original local processor directory. A completed run with no steps remaining cannot be resumed to extend the schedule: changing total steps is rejected by the checkpoint contract.

## Common failures

| Symptom | Check |
| :--- | :--- |
| Output directory already exists | Choose a fresh directory; existing runs are never overwritten. |
| Missing local model/config | Download model and processor first; the trainer uses `local_files_only=True`. |
| Duplicate images/answers or mixed family branches | Fix the family contract rather than sampling away members. |
| Sequence exceeds `--max-length` | Inspect image resolution and answer length; do not truncate answer supervision. |
| Replay score drift | Check stochastic layers, determinism, and accidental parameter updates between passes. |
| CUDA or host-memory OOM | Measure real peak memory and family size; CPU offload does not eliminate RAM needs. |
| Resume contract mismatch | Restore the original data, arguments, and scheduler horizon. |
