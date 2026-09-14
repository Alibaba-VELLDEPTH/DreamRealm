"""Strict full-answer teacher forcing; no truncation and no candidate leakage."""
from pathlib import Path
import torch
from PIL import Image
from .prepare import validate_family


class FamilyEncoder:
    def __init__(self, processor, max_length=6144, media_root=None):
        self.processor = processor
        self.max_length = max_length
        self.media_root = media_root

    def image_path(self, path):
        prefix = '/data/oss_bucket_0/deltalake/other_sample/'
        if self.media_root and path.startswith(prefix):
            return str(Path(self.media_root) / path[len(prefix):])
        return path

    def encode(self, family):
        validate_family(family)
        views = [dict(image=family['original_image'], answer=family['original_answer'])] + family['edited_pairs']
        answers = [v['answer'] for v in views] if family['edited_pairs'] else [family['original_answer']] + family['original_negative_answers']
        cells, targets = [], {}
        for i, view in enumerate(views):
            with Image.open(self.image_path(view['image'])) as im:
                image = im.convert('RGB')
            messages = [dict(role='user', content=[dict(type='image', image=image), dict(type='text', text=family['question'])])]
            prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
            prompt_encoded = self.processor(text=[prompt], images=[image], return_tensors='pt')
            prompt_ids = prompt_encoded['input_ids'][0]
            for j, answer in enumerate(answers):
                full = self.processor.apply_chat_template(messages + [dict(role='assistant', content=answer)], tokenize=False, add_generation_prompt=False, enable_thinking=False)
                if not full.startswith(prompt):
                    raise ValueError('Full chat template does not extend non-thinking prompt')
                encoded = self.processor(text=[full], images=[image], return_tensors='pt')
                ids = encoded['input_ids'][0]
                start = len(prompt_ids)
                if not torch.equal(ids[:start], prompt_ids):
                    raise ValueError('Prompt/answer token boundary changed; refusing ambiguous supervision')
                if len(ids) > self.max_length:
                    raise ValueError(f"{family['family_id']} cell ({i},{j}) has {len(ids)} tokens > max_length={self.max_length}; truncation forbidden")
                # Include the chat EOS, exclude any trailing separator/newline.
                eos = self.processor.tokenizer.eos_token_id
                endings = (ids[start:] == eos).nonzero().flatten()
                if not len(endings):
                    raise ValueError('Assistant turn has no EOS')
                end = start + int(endings[0]) + 1
                if end <= start + 1:
                    raise ValueError('Empty scored answer')
                labels = torch.full_like(ids, -100)
                labels[start:end] = ids[start:end]
                target = labels[labels != -100]
                decoded = self.processor.tokenizer.decode(
                    target[:-1], skip_special_tokens=False, clean_up_tokenization_spaces=False
                )
                if decoded != answer:
                    raise ValueError('Scored tokens do not decode to exactly the supplied answer plus EOS')
                if j in targets and not torch.equal(target, targets[j]):
                    raise ValueError('Same answer has different target tokens/masks across images')
                targets[j] = target.clone()
                cells.append(dict(inputs=dict(encoded), labels=labels.unsqueeze(0), row=i, col=j))
        return dict(family_id=family['family_id'], with_edit=bool(family['edited_pairs']),
                    shape=(len(views), len(answers)), cells=cells,
                    target_token_counts=[len(targets[j]) for j in range(len(answers))])


def score_cell(model, cell, device):
    inputs = {k:v.to(device) for k,v in cell['inputs'].items()}
    labels = cell['labels'].to(device)
    outputs = model(**inputs, use_cache=False)
    # Only materialize FP32 logits at assistant target positions, not image/user tokens.
    target = labels[:, 1:]
    mask = target != -100
    selected = outputs.logits[:, :-1][mask].float()
    target = target[mask]
    if target.numel() == 0:
        raise ValueError('No answer token after causal shift')
    return -torch.nn.functional.cross_entropy(selected, target, reduction='mean')
