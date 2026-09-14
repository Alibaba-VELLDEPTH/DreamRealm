# DreamRealm Qwen3.5-VL Training

Private research repository for the RealmForge family-matrix training used by the
Qwen3.5-VL 9B experiment.

## Reproduced run

- Model family: Qwen3.5-VL 9B
- Objective: answer-only family SFT with row/column cycle preference terms
- Training families: 36,800
- Distributed setup: 1 worker, 8 GPUs, DeepSpeed ZeRO-3
- Optimizer steps: 1,383 (461 per epoch, 3 epochs)
- Checkpoints: 461, 922, 1,383
- Verified result: `ZERO3_FAMILY_FORMAL8_PASS`
- Final training loss: 0.6693

The full dataset and model checkpoints are intentionally not stored in Git.
Their internal OSS locations and hashes are documented in `DATA_CHANGE.md` and
the launch script.

## Main entry points

- `train_dreamrealm_zero3.py`: Qwen3.5 family-matrix ZeRO-3 trainer.
- `dreamrealm/`: matrix encoding, objectives, batching, distributed helpers, and
  checkpoint persistence.
- `run_nebula_zero3_family_formal8_r06.sh`: exact 8-GPU smoke-gate plus formal
  training workflow used for the successful run.
- `submission/submit_realmforge_8g.py`: fixed-pool Nebula admission and submission
  helper. Credentials are read from environment variables or an explicit local
  OSS config file and are never committed.
- `tests/`: unit and dry-run coverage.

## Safety and reproducibility

The launch path verifies the runtime archive and dataset SHA-256 before training,
runs an 8-rank smoke gate, starts formal training from the original weights, and
validates the final checkpoint at step 1,383. Resource admission is fixed to the
guaranteed `secureagi_common_l20z_cn1040` pool with 8 L20Z GPUs.

Before any new formal Nebula submission, re-check the pool, quota group, device
type, GPU count, code/data hashes, expected duration, validation plan, and output
prefix. Do not stop or migrate an existing run without explicit authorization.

## Local checks

```bash
python -m compileall dreamrealm joint_dpo train_dreamrealm_zero3.py
pytest -q
bash -n run_nebula_zero3_family_formal8_r06.sh
```

See `README_DREAMREALM.md`, `docs/DreamRealm_训练方法说明.md`, and
`docs/data-format.md` for the method and data schema.
