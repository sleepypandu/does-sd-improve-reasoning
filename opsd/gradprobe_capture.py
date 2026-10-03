"""Frozen-model gradient capture for the OPSD, GRPO and SFT trainers.

`attach(trainer, out_path, iterations)` turns a training run into a gradient probe. The learning
rate is forced to 0, so every iteration differentiates the same base weights, and at each of the
first `iterations` optimizer steps the full gradient (the mean over the accumulation window, after
DeepSpeed has clipped it to the run's `max_grad_norm`) is added to an fp32 buffer on rank 0. Then
training stops and the mean over iterations is written to `out_path` as safetensors, with its
metadata also in `<out_path>.meta.json`. Data, decoding, loss and clipping are the training run's.

A parameter fingerprint taken before the first iteration and after the last checks that no weight
moved.
"""
import json
import os

import torch
from transformers import TrainerCallback


def _uses_deepspeed(trainer):
    acc = trainer.accelerator
    return getattr(acc, "deepspeed_engine_wrapped", None) is not None or str(acc.distributed_type).endswith(
        "DEEPSPEED"
    )


def _full_grad(p, use_deepspeed):
    # ZeRO-2 and ZeRO-3 both partition gradients, so p.grad may be a shard (or None). safe_get_full_grad
    # reassembles the full gradient; it is a collective, so every rank must call it for every parameter.
    if use_deepspeed:
        from deepspeed.utils import safe_get_full_grad

        g = safe_get_full_grad(p)
    else:
        g = p.grad
    return None if g is None else g.detach().to(torch.float32)


def _fingerprint(model):
    """(float64 sum of every parameter, number of parameters covered).

    Under ZeRO-3 an ungathered p.data is an empty placeholder, so the rank's persistent shard
    (ds_tensor) is summed instead.
    """
    total, n = torch.zeros((), dtype=torch.float64), 0
    with torch.no_grad():
        for _, p in sorted(model.named_parameters()):
            t = getattr(p, "ds_tensor", None)
            t = p.detach() if t is None else t
            if t.numel() == 0:
                continue
            total += t.to(torch.float64).sum().cpu()
            n += 1
    return total.item(), n


class GradientProbe(TrainerCallback):
    def __init__(self, trainer, out_path, iterations, meta=None):
        self.trainer = trainer
        self.out_path = out_path
        self.iterations = iterations
        self.meta = dict(meta or {})
        self.sum = {}
        self.n = 0
        self.norms = []
        self.preclip_norms = []

    def is_main(self):
        return self.trainer.args.process_index == 0

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        if args.learning_rate != 0:
            raise ValueError(f"a gradient probe needs learning_rate 0, got {args.learning_rate}")
        if state.max_steps <= self.iterations:
            # The trainer saves a checkpoint when it reaches max_steps; the probe must stop first.
            raise ValueError(f"max_steps ({state.max_steps}) must exceed the probe's {self.iterations} iterations")
        self.model = model
        self.use_deepspeed = _uses_deepspeed(self.trainer)
        wrapped = getattr(self.trainer.accelerator, "deepspeed_engine_wrapped", None)
        self.engine = getattr(wrapped, "engine", None)
        self.fingerprint = _fingerprint(model)
        if self.fingerprint[1] == 0:
            raise RuntimeError("the weight fingerprint covers no parameters")
        clip = getattr(getattr(self.engine, "optimizer", None), "clip_grad", None)
        self.meta.update(
            backend="deepspeed" if self.use_deepspeed else "torch",
            clip_grad=repr(float(clip if clip is not None else args.max_grad_norm)),
        )

    def on_pre_optimizer_step(self, args, state, control, model=None, **kwargs):
        keep = self.is_main()
        with torch.no_grad():
            grads = {}
            for name, p in self.model.named_parameters():
                g = _full_grad(p, self.use_deepspeed)
                if g is not None and keep:
                    grads[name] = g.cpu()
                p.grad = None
            if keep:
                if not grads:
                    raise RuntimeError("no parameter has a gradient at the optimizer step")
                for name, g in grads.items():
                    if name in self.sum:
                        self.sum[name] += g
                    else:
                        self.sum[name] = g.clone()
                self.norms.append(float(sum(g.double().pow(2).sum() for g in grads.values()) ** 0.5))
                if self.engine is not None:
                    preclip = self.engine.get_global_grad_norm()
                    self.preclip_norms.append(None if preclip is None else float(preclip))
        self.n += 1
        if self.n >= self.iterations:
            control.should_training_stop = True

    def on_train_end(self, args, state, control, model=None, **kwargs):
        after = _fingerprint(self.model)
        if after != self.fingerprint:
            raise RuntimeError(f"weights moved during the probe: fingerprint {self.fingerprint} -> {after}")
        if self.is_main():
            self.save()

    def save(self):
        from safetensors.torch import save_file

        if self.n != self.iterations:
            raise RuntimeError(f"captured {self.n} iterations, expected {self.iterations}")
        mean = {name: g / self.n for name, g in self.sum.items()}
        norm = float(sum(g.double().pow(2).sum() for g in mean.values()) ** 0.5)
        if norm == 0:
            raise RuntimeError("the mean gradient is exactly zero")
        meta = dict(
            self.meta,
            iterations=str(self.n),
            global_norm=repr(norm),
            iteration_norms=json.dumps(self.norms),
            preclip_norms=json.dumps(self.preclip_norms),
            fingerprint=repr(self.fingerprint[0]),
            fingerprint_params=str(self.fingerprint[1]),
            n_tensors=str(len(mean)),
            n_elements=str(sum(g.numel() for g in mean.values())),
        )
        os.makedirs(os.path.dirname(os.path.abspath(self.out_path)), exist_ok=True)
        save_file(mean, self.out_path, metadata=meta)
        with open(self.out_path + ".meta.json", "w") as f:
            json.dump(meta, f, indent=2, sort_keys=True)


def attach(trainer, out_path, iterations, meta=None):
    """Make `trainer.train()` record a gradient instead of training. out_path 'auto' is
    <output_dir>/gradient.safetensors."""
    trainer.args.learning_rate = 0.0
    if out_path == "auto":
        out_path = os.path.join(trainer.args.output_dir, "gradient.safetensors")
    probe = GradientProbe(trainer, out_path, iterations, meta)
    trainer.add_callback(probe)
    return probe
