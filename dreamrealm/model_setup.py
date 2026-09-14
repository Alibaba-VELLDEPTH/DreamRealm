"""Model setup used by the local trainer."""


def freeze_vision_backbone(model):
    """Freeze image patch/block weights while training language and projection layers."""
    trainable = []
    frozen_markers = ("visual.patch_embed", "visual.blocks")
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(not any(marker in name for marker in frozen_markers))
        if parameter.requires_grad:
            trainable.append(name)
    if not trainable:
        raise ValueError("No trainable parameters remain after freezing the vision backbone")
    return trainable
