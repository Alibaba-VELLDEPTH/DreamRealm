"""Exact first-order chain rule over a whole score matrix with bounded GPU graphs."""
import torch
from .objective import family_loss


def backward_family(encoded, score_fn, scale=1.0, tau=1.0, row_weight=0.5, col_weight=0.5, replay_atol=2e-4):
    # No parameter updates or random dropout are permitted between these passes.
    with torch.no_grad():
        values = torch.stack([score_fn(c).detach().float() for c in encoded['cells']]).reshape(encoded['shape'])
    scores = values.detach().requires_grad_(True)
    loss, parts = family_loss(scores, encoded['with_edit'], tau, row_weight, col_weight)
    if not torch.isfinite(loss):
        raise FloatingPointError('Non-finite family loss')
    coefficients, = torch.autograd.grad(loss, scores)
    max_error = 0.0
    for cell, coefficient, original in zip(encoded['cells'], coefficients.flatten(), values.flatten()):
        score = score_fn(cell)
        error = float((score.detach().float()-original).abs())
        max_error = max(max_error, error)
        if error > replay_atol:
            raise RuntimeError(f'Replayed score drift {error}; stochastic scoring would change the objective')
        (score * coefficient.detach() * scale).backward()
    n, m = encoded['shape']
    result = dict(family_id=encoded['family_id'], branch='edited' if encoded['with_edit'] else 'noedit',
                  shape=[n,m], scores=values.cpu().tolist(), loss=float(loss.detach()),
                  **{k:float(v) for k,v in parts.items()}, replay_max_abs_error=max_error,
                  target_token_counts=encoded.get('target_token_counts', []), gradient_scale=scale)
    return result


class CPUAdamW:
    """FP32 master weights and Adam states on CPU; model/gradients stay on GPU.

    Intended for the physical-machine single-GPU path, not a distributed wrapper.
    A complete family/batch backward finishes before step() copies any new weights.
    """
    def __init__(self, model, lr, weight_decay=0.0):
        self.named = [(n,p) for n,p in model.named_parameters() if p.requires_grad]
        self.masters = [torch.nn.Parameter(p.detach().to(device='cpu', dtype=torch.float32).clone()) for _,p in self.named]
        self.optimizer = torch.optim.AdamW(self.masters, lr=lr, weight_decay=weight_decay, foreach=False)

    def zero_grad(self):
        for _,p in self.named: p.grad = None
        self.optimizer.zero_grad(set_to_none=True)

    def step(self):
        for (_,p), master in zip(self.named, self.masters):
            master.grad = None if p.grad is None else p.grad.detach().to(device='cpu', dtype=torch.float32)
            p.grad = None
        self.optimizer.step()
        with torch.no_grad():
            for (_,p), master in zip(self.named, self.masters):
                p.copy_(master.to(device=p.device, dtype=p.dtype))
        self.optimizer.zero_grad(set_to_none=True)

    def state_dict(self):
        return dict(optimizer=self.optimizer.state_dict(), masters=[x.detach() for x in self.masters])

    def load_state_dict(self, state):
        if len(state['masters']) != len(self.masters): raise ValueError('Optimizer/model mismatch')
        with torch.no_grad():
            for saved, master, (_,p) in zip(state['masters'],self.masters,self.named):
                master.copy_(saved); p.copy_(master.to(device=p.device,dtype=p.dtype))
        self.optimizer.load_state_dict(state['optimizer'])
