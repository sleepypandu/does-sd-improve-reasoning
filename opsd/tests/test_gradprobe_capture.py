import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import load_file
from transformers import TrainerControl, TrainerState

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gradprobe_capture import attach  # noqa: E402


def make(tmp_path, iterations=2, max_steps=3):
    model = torch.nn.Linear(3, 2)
    args = SimpleNamespace(learning_rate=5e-6, max_grad_norm=1.0, process_index=0, output_dir=str(tmp_path))
    trainer = SimpleNamespace(args=args, accelerator=SimpleNamespace(distributed_type="NO"), callbacks=[])
    trainer.add_callback = trainer.callbacks.append
    probe = attach(trainer, "auto", iterations)
    state = TrainerState()
    state.max_steps = max_steps
    return model, args, probe, state


def step(model, probe, args, state, control, x):
    model.zero_grad()
    model(x).sum().backward()
    probe.on_pre_optimizer_step(args, state, control, model=model)


def test_records_the_mean_gradient(tmp_path):
    model, args, probe, state = make(tmp_path)
    assert args.learning_rate == 0
    control = TrainerControl()
    probe.on_train_begin(args, state, control, model=model)
    xs = [torch.randn(4, 3), torch.randn(4, 3)]
    expected = {}
    for x in xs:
        model.zero_grad()
        model(x).sum().backward()
        for name, p in model.named_parameters():
            expected[name] = expected.get(name, 0) + p.grad / len(xs)
    for i, x in enumerate(xs):
        assert not control.should_training_stop
        step(model, probe, args, state, control, x)
    assert control.should_training_stop
    probe.on_train_end(args, state, control, model=model)
    saved = load_file(str(tmp_path / "gradient.safetensors"))
    for name, g in expected.items():
        torch.testing.assert_close(saved[name], g)
    assert (tmp_path / "gradient.safetensors.meta.json").exists()


def test_refuses_a_learning_rate(tmp_path):
    model, args, probe, state = make(tmp_path)
    args.learning_rate = 1e-6
    with pytest.raises(ValueError):
        probe.on_train_begin(args, state, TrainerControl(), model=model)


def test_refuses_to_reach_max_steps(tmp_path):
    model, args, probe, state = make(tmp_path, max_steps=2)
    with pytest.raises(ValueError):
        probe.on_train_begin(args, state, TrainerControl(), model=model)


def test_detects_moved_weights(tmp_path):
    model, args, probe, state = make(tmp_path)
    control = TrainerControl()
    probe.on_train_begin(args, state, control, model=model)
    for _ in range(2):
        step(model, probe, args, state, control, torch.randn(4, 3))
    with torch.no_grad():
        model.bias += 1
    with pytest.raises(RuntimeError, match="moved"):
        probe.on_train_end(args, state, control, model=model)
