# Gradient similarity

Code for the gradient-similarity analysis (Section 4.4, Appendix E.3, Tables 3, 4 and 18-21): at
the frozen base model, how well does the gradient of each training objective align with the RL
gradient on held-out problems?

| path | contents |
|---|---|
| `opsd/gradprobe_capture.py` | gradient capture for the OPSD, GRPO and SFT trainers (`--gradprobe_out`) |
| `sdpo/verl/gradprobe.py` | gradient capture for the verl FSDP actor (`GRADPROBE_OUT`) |
| `scripts/gradprobe/` | reference sets, recording launchers, analysis launcher |
| `gradprobe/` | the analysis: pooled cosine, jackknife SE, gradient noise scale |

All commands run from the repository root, with the environment of the experiment family
(`env/opsd` for math, see [opsd.md](opsd.md); `env/sdpo` for LiveCodeBench, see
[sdpo_lcb.md](sdpo_lcb.md)). Recording takes one node with 8 H100s per run; the analysis is
CPU-only.

## What is recorded

A recording is one run of the paper's training launcher with learning rate 0. Data, rollouts,
loss, batch size and gradient clipping are those of the training run, but every step
differentiates the same base weights. One *iteration* is one training step's gradient:

- **math** (TRL + DeepSpeed): one optimizer step, i.e. the gradient of one 32-sequence batch
  (GRPO: 4 prompts x 8 rollouts) after clipping to the run's `max_grad_norm` (0.1 for OPSD, 1.0
  for GRPO and SFT). The full gradient is reassembled from the ZeRO partitions on every rank and
  summed on rank 0, which writes the mean over 64 iterations to `gradient.safetensors` (with
  `gradient.safetensors.meta.json`). GRPO uses one update per generation batch
  (`--num_iterations 1`; training uses 2, whose second update would repeat the gradient at
  learning rate 0).
- **LiveCodeBench** (verl + FSDP): one generation batch of 32 prompts x 8 rollouts, whose
  gradient is the sum of its clipped mini-batch steps (4 for GRPO, 32 for SDPO), i.e. what one
  training step moves the weights by. Each rank writes the mean over 8 iterations of its own FSDP
  shard to `gradient_rank<k>.safetensors`; shards are never reassembled.

Gradients are dense (all parameters, no LoRA) and fp32. Both probes refuse to run unless the
learning rate is 0 and compare a fingerprint of the weights before the first and after the last
iteration.

The reference is GRPO on held-out problems: AIME24 + AIME25 + HMMT25 (90 problems, thinking
off) for math, the 66 held-out `test_unseen` problems for LiveCodeBench. The methods train on
their usual data: OpenThoughts for math (GRPO with and without thinking, SFT, and OPSD for
teacher thinking on/off x $z$ in {reference, none, shuffled}; Olmo-3-7B-Instruct has a single
GRPO arm and its thinking teacher is Olmo-3-7B-Think), and the 65 training problems for
LiveCodeBench (GRPO and SDPO). Every arm is recorded for seeds 0-3.

## Recording

```bash
bash scripts/gradprobe/build_reference_set.sh             # data/gradprobe/aime24_aime25_hmmt25.parquet
MODEL=Qwen/Qwen3-8B SEEDS="0 1 2 3" bash scripts/gradprobe/math.sh

bash scripts/sdpo_lcb/prepare_data.sh                     # data/sdpo_lcb/lcb_exp
MODEL=Qwen/Qwen3-8B SEEDS="0 1 2 3" bash scripts/gradprobe/lcb.sh
```

`math.sh` calls `scripts/opsd/train_{grpo,sft,opsd}.sh` with the per-model presets of
`scripts/opsd/presets.sh` and `--learning_rate 0 --use_peft False --gradprobe_out auto
--gradprobe_iterations 64`, and `max_steps` one past the last iteration so the trainer never
reaches its end-of-run save. `lcb.sh` calls `scripts/sdpo_lcb/train.sh` with
`actor_rollout_ref.actor.optim.lr=0`, `STEPS=8`, and checkpoints and validation off. Both run the
arms one after another; `ARMS` (`reference grpo sft opsd` / `reference grpo sdpo`), `SEEDS`,
`GRPO_THINKING`, `TEACHER_THINKING` and `Z` (math) select a subset, e.g. to spread a grid over
nodes, `ITERATIONS` changes the number of iterations, and `DRY_RUN=1` prints the commands. Gradients land in

```
outputs/gradprobe/math/<model>/{reference,train}/<run>/gradient.safetensors
outputs/gradprobe/lcb/<model>/{reference,train}/<run>/gradient_rank<k>.safetensors
```

with the run names of the training launchers (`OUT_ROOT` moves the tree).

Disk: one gradient is 4 bytes per parameter, about 6.9 / 16 / 33 / 59 GB for Qwen3-1.7B / 4B /
8B / 14B and 29 GB for Olmo-3-7B-Instruct (an 8B LiveCodeBench run is 8 shards of 4.1 GB). The
Qwen3-8B math grid is 40 runs (1.3 TB), an LCB grid 12 runs (0.4 TB at 8B). Rank 0 of a math
run holds the running sum and one iteration's gradient in host memory, about twice the gradient
size.

## Analysis

```bash
FAMILY=math MODEL=Qwen/Qwen3-8B NPARTS=8 bash scripts/gradprobe/analyze.sh
FAMILY=lcb  MODEL=Qwen/Qwen3-8B NPARTS=8 bash scripts/gradprobe/analyze.sh
```

prints, per arm, the pooled cosine to the reference with its jackknife standard error and the
gradient noise scale, and writes them to `<model>/analysis/report.json`. The math arms are
`GRPO_think`, `GRPO_nothink`, `SFT` and `OPSD_<think|nothink>_<ref|none|shuffled>` (teacher
thinking, $z$); for Olmo-3-7B-Instruct, `GRPO_think` is its single GRPO arm and `OPSD_think_*`
the Olmo-3-7B-Think teacher. The LiveCodeBench arms are `GRPO` and `SDPO`. With $g_s$ an arm's
gradient for seed $s$ (a mean over iterations), $S$ seeds and $\bar g = \frac1S\sum_s g_s$:

- pooled cosine: $\cos(\bar g_\text{ref}, \bar g_\text{method})$;
- jackknife SE: $\sqrt{\frac{S-1}{S}\sum_s (c_{-s} - \bar c)^2}$, where $c_{-s}$ is the pooled
  cosine with seed $s$ dropped from the reference and the method together;
- gradient noise scale: $\mathrm{tr}(\Sigma)/\lVert\bar g\rVert^2$ with
  $\mathrm{tr}(\Sigma) = \frac{1}{S-1}\sum_s \lVert g_s - \bar g\rVert^2$, over the seed
  means.

All three are functions of the Gram matrix of the gradients, which `gradprobe/analyze.py gram`
accumulates in float64 one bounded slice of one tensor at a time (single files and per-rank shards
alike), in `NPARTS` processes over disjoint tensors; memory stays at a few GB per process. Reading
is I/O-bound: with 8 processes, the Qwen3-8B math grid (1.3 TB) took about 35 minutes on our shared
file system and a Qwen3-8B LiveCodeBench grid (0.4 TB) about 3. For other layouts,
call it directly:

```bash
python -m gradprobe.analyze gram --reference 'ref/*-seed-*' --method 'SFT=train/sft-*-seed-*' --out gram.npz
python -m gradprobe.analyze report gram.npz --json report.json
```

Each glob matches one gradient per seed (the seed is read from `seed-<n>` or `seed<n>` in the
path); the reference and every method must have the same seeds. Unit tests (CPU):
`python -m pytest gradprobe opsd/tests/test_gradprobe_capture.py`.

| paper | command |
|---|---|
| Table 3 (Qwen3-8B, math) | `FAMILY=math MODEL=Qwen/Qwen3-8B`, column `cosine`/`SE` |
| Table 18 (Qwen3-14B, math) | `FAMILY=math MODEL=Qwen/Qwen3-14B` |
| Table 19 (Olmo-3-7B-Instruct, math) | `FAMILY=math MODEL=allenai/Olmo-3-7B-Instruct` |
| Table 4 (LiveCodeBench) | `FAMILY=lcb` for `Qwen/Qwen3-8B`, `Qwen/Qwen3-14B`, `allenai/Olmo-3-7B-Instruct` |
| Table 20 (noise scale, math) | `FAMILY=math MODEL=Qwen/Qwen3-8B`, column `noise scale` |
| Table 21 (noise scale, LiveCodeBench) | `FAMILY=lcb MODEL=Qwen/Qwen3-8B`, column `noise scale` |
