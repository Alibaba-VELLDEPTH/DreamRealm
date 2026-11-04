# Image inference

Model: [Alibaba-VELLDEPTH/Qwen35-9b-DreamRealm](https://huggingface.co/Alibaba-VELLDEPTH/Qwen35-9b-DreamRealm), based on [Qwen/Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B).

Install the repository's dependencies and download the complete model repository, not only the weight file:

```bash
python -m pip install -r requirements.txt
python -m pip install huggingface_hub
hf download Alibaba-VELLDEPTH/Qwen35-9b-DreamRealm \
  --local-dir /models/Qwen35-9b-DreamRealm
```

The release configuration records Transformers 5.3.0. The following example follows the Qwen3.5 processor/generation interface and disables thinking to match the training prompt mode. It requires CUDA and sufficient memory; it is a usage example, not a measured GPU compatibility guarantee.

```python
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

model_path = "/models/Qwen35-9b-DreamRealm"
processor = AutoProcessor.from_pretrained(
    model_path, local_files_only=True, max_pixels=262144
)
model = Qwen3_5ForConditionalGeneration.from_pretrained(
    model_path,
    local_files_only=True,
    dtype=torch.bfloat16,
    attn_implementation="sdpa",
).to("cuda").eval()

with Image.open(Path("/path/to/your/image.jpg")) as source:
    image = source.convert("RGB")

messages = [{
    "role": "user",
    "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": "What is happening in the image?"},
    ],
}]
prompt = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
)
inputs = processor(text=[prompt], images=[image], return_tensors="pt").to("cuda")
with torch.inference_mode():
    output = model.generate(**inputs, max_new_tokens=128, do_sample=False)
answer_ids = output[:, inputs["input_ids"].shape[1]:]
print(processor.batch_decode(answer_ids, skip_special_tokens=True)[0])
```

Keep the checkpoint's processor and chat template together with the model. Avoid claiming a benchmark result from this free-form example: benchmark prompts, answer parsing, image settings, and evaluation protocols must be controlled separately.

Upstream reference: [official Qwen3.5-9B model card](https://huggingface.co/Qwen/Qwen3.5-9B).
