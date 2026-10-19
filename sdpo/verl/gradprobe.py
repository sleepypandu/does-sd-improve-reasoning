"""Frozen-model gradient capture for the FSDP actor, enabled by GRADPROBE_OUT=<dir>.

One iteration is one generation batch (one `update_policy` call); its gradient is the sum of the
clipped gradients of the batch's mini-batch optimizer steps, i.e. what one training step would
move the weights by. The learning rate must be 0, so every iteration sees the base weights. After
GRADPROBE_ITERS iterations each rank writes the mean of its own FSDP gradient shard to
<dir>/gradient_rank<k>.safetensors (dot products are sums over ranks, so shards are never
reassembled) and rank 0 writes <dir>/gradprobe_meta.json. A parameter fingerprint taken before the
first optimizer step checks that no weight moved.
"""
import json
import os

import torch


def _fingerprint(module):
    with torch.no_grad():
        return sum(p.detach().to(torch.float64).sum().item() for _, p in sorted(module.named_parameters()))


class GradProbe:
    def __init__(self, out_dir, iterations):
        self.out_dir = out_dir
        self.iterations = iterations
        self.rank = int(os.environ.get("RANK", "0"))
        self.world_size = int(os.environ.get("WORLD_SIZE", "1"))
        self.sum = {}
        self.n = 0
        self.steps = 0
        self.preclip_norms = []
        self.fingerprint = None

    def on_optimizer_step(self, module, optimizer, grad_norm):
        """After the gradient is clipped, before the optimizer step."""
        if self.fingerprint is None:
            lrs = [group["lr"] for group in optimizer.param_groups]
            if any(lr != 0 for lr in lrs):
                raise ValueError(f"a gradient probe needs actor_rollout_ref.actor.optim.lr=0, got {lrs}")
            self.fingerprint = _fingerprint(module)
        with torch.no_grad():
            for name, p in module.named_parameters():
                if p.grad is None:
                    continue
                g = p.grad.detach().to(torch.float32).cpu()
                if name in self.sum:
                    self.sum[name] += g
                else:
                    self.sum[name] = g.clone()
        self.steps += 1
        self.preclip_norms.append((self.n, float(grad_norm)))

    def on_update_end(self, module):
        """At the end of update_policy: one generation batch is one iteration."""
        if self.steps == 0 or self.n >= self.iterations:
            return
        self.n += 1
        if self.n == self.iterations:
            self.save(module)

    def save(self, module):
        from safetensors.torch import save_file

        after = _fingerprint(module)
        if after != self.fingerprint:
            raise RuntimeError(f"weights moved during the probe on rank {self.rank}: {self.fingerprint} -> {after}")
        mean = {name: g / self.n for name, g in self.sum.items()}
        meta = dict(
            rank=str(self.rank),
            world_size=str(self.world_size),
            iterations=str(self.n),
            optimizer_steps=str(self.steps),
            fingerprint=repr(self.fingerprint),
            n_tensors=str(len(mean)),
            n_elements=str(sum(g.numel() for g in mean.values())),
        )
        os.makedirs(self.out_dir, exist_ok=True)
        save_file(mean, os.path.join(self.out_dir, f"gradient_rank{self.rank}.safetensors"), metadata=meta)
        if self.rank == 0:
            with open(os.path.join(self.out_dir, "gradprobe_meta.json"), "w") as f:
                json.dump(dict(meta, preclip_norms=self.preclip_norms), f, indent=2)
        print(f"[gradprobe] rank {self.rank}: wrote {self.n} iterations ({self.steps} optimizer steps)", flush=True)


_probe = None


def get():
    """The process's probe, or None unless GRADPROBE_OUT is set."""
    global _probe
    if _probe is None and os.environ.get("GRADPROBE_OUT"):
        _probe = GradProbe(os.environ["GRADPROBE_OUT"], int(os.environ.get("GRADPROBE_ITERS", "8")))
    return _probe
