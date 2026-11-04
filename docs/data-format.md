# Family data format

Input is UTF-8 JSONL: **one non-empty line per complete family**. Images may use absolute paths or paths relative to `--media-root`. The examples below illustrate the schema only; they are not a released dataset.

## Edited family: N images × N answers

An original image/answer plus one or more edited image/answer pairs forms a square score matrix. Every answer is scored against every image; matched pairs occupy its diagonal.

```json
{
  "family_id": "clock-001",
  "question": "What time is shown on the clock?",
  "original_image": "images/clock-original.jpg",
  "original_answer": "12:05",
  "edited_pairs": [
    {"image": "images/clock-edited-1.jpg", "answer": "12:15"},
    {"image": "images/clock-edited-2.jpg", "answer": "12:25"}
  ],
  "original_negative_answers": []
}
```

This example produces a `3 × 3` matrix. Do not split the family or add unmatched original-image negative-answer columns to this branch.

## No-edit family: 1 image × (M+1) answers

If no edited image is available, retain the original correct answer and zero or more distinct negative answers.

```json
{
  "family_id": "sign-001",
  "question": "What number is on the sign?",
  "original_image": "images/sign.jpg",
  "original_answer": "42",
  "edited_pairs": [],
  "original_negative_answers": ["24", "43"]
}
```

This example produces a `1 × 3` matrix. An empty negative list is valid and reduces this branch to correct-answer SFT.

## Validation rules

- All six fields above are required.
- `family_id`, `question`, `original_image`, and `original_answer` must be non-empty strings. Family IDs must be unique in a training file.
- Each edited pair provides an image path and non-empty answer. Image paths within a family must be unique; answer strings must be unique after whitespace normalization.
- No-edit negative answers must be non-empty, distinct after whitespace normalization, and different from the correct answer.
- Edited families must leave `original_negative_answers` empty.
- Image availability and tokenizer compatibility are checked by real-processor preflight, not by structural validation alone.
- The encoder scores the exact supplied answer plus EOS. If any cell exceeds `--max-length`, it fails instead of truncating. Keep the checkpoint's chat template and tokenizer consistent.

Store each object on **one line** in the actual JSONL file; the examples above are expanded only for readability.

## Legacy pair conversion

`dreamrealm.prepare` also contains a converter for an earlier R4 `sft` / `cycle_pair` record format. This is a compatibility utility, not an arbitrary pair-to-family converter or a public data-construction pipeline. It requires consistent family positions, declared sizes, questions, and source IDs (with exactly one original member). Its diagnostic fixture selection expects accepted edited families of sizes two and three and at least one no-edit family. Use the native family format above for new datasets.

See [training](training.md) for processor preflight and [reproducibility](reproducibility.md) for experiment-record requirements.
