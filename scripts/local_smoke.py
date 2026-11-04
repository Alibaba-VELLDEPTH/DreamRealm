"""Fast CPU smoke test for the data contract and exact family-gradient path."""

from pathlib import Path
import json
import sys

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dreamrealm.engine import backward_family  # noqa: E402
from dreamrealm.prepare import validate_family  # noqa: E402


def main():
    family = validate_family(
        {
            "family_id": "local-smoke",
            "question": "Which answer belongs to each image?",
            "original_image": "images/original.jpg",
            "original_answer": "original",
            "edited_pairs": [{"image": "images/edited.jpg", "answer": "edited"}],
            "original_negative_answers": [],
        }
    )

    torch.manual_seed(7)
    model = torch.nn.Linear(4, 1)
    features = torch.randn(4, 4)
    encoded = {
        "family_id": family["family_id"],
        "with_edit": True,
        "shape": (2, 2),
        "cells": list(features),
        "target_token_counts": [1, 1],
    }
    metrics = backward_family(encoded, lambda cell: model(cell).squeeze())
    grad_norm = torch.linalg.vector_norm(
        torch.cat([parameter.grad.flatten() for parameter in model.parameters()])
    )
    if not torch.isfinite(grad_norm) or float(grad_norm) <= 0:
        raise RuntimeError("Smoke test produced an invalid gradient")

    print(
        json.dumps(
            {
                "status": "PASS",
                "branch": metrics["branch"],
                "shape": metrics["shape"],
                "loss": metrics["loss"],
                "gradient_norm": float(grad_norm),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
