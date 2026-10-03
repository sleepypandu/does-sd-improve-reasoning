# SDPO vs. GRPO vs. SFT on SciKnowEval and ToolAlpaca

Code for the science and tool-use comparison (Section 4.2, Appendix E.1, Figure 4, Table 7).
Qwen3-8B with thinking off is trained for 200 steps by GRPO, SDPO, or final-answer SFT on
SciKnowEval biology/chemistry/material/physics and on ToolAlpaca. Each method/dataset pair runs
with 4 seeds, one 8-GPU node per run. The training code is the SDPO verl fork in `sdpo/`.

## Environment

These experiments share the pinned SDPO environment with the LiveCodeBench runs:

```bash
bash env/sdpo/setup_env.sh      # creates env/sdpo/.venv
source env/sdpo/.venv/bin/activate
```

All scripts resolve `MODEL` (default `Qwen/Qwen3-8B`) to a local snapshot directory before they
launch, because Ray workers and torchrun ranks would otherwise query the Hub. On an offline node,
populate the Hugging Face cache first and set `HF_HUB_OFFLINE=1`.

## Data

```bash
bash scripts/sdpo_sci/prepare_data.sh
```

For each dataset, this writes `{train,test}.parquet` (RL, `data/preprocess.py`) and
`{train,test}.sft.parquet` (SFT, `data/preprocess_sft.py`) next to the committed JSON in
`sdpo/datasets/`. The output is deterministic.

- **SciKnowEval splits:** these are the committed upstream splits. `data/split_tasks.py
  --test_ratio 0.1 --seed 42` reproduces them byte for byte (`RESPLIT=1` reruns it).
- **ToolAlpaca:** this uses the upstream `datasets/tooluse` split.
- **SFT targets:** an SFT row has the same system and user messages as the RL prompt. The target is
  `<answer>{letter}</answer>` for SciKnowEval, and one `Action: …\nAction Input: {json}` block per
  ground-truth call for ToolAlpaca.

## Training

```bash
METHOD=sdpo DATASET=chemistry SEED=1 bash scripts/sdpo_sci/train_rl.sh   # METHOD=sdpo|grpo
DATASET=biology SEED=1 bash scripts/sdpo_sci/train_sft.sh
DRY_RUN=1 bash scripts/sdpo_sci/paper_grid.sh                           # 3 methods x 5 datasets x seeds 1-4
```

`DATASET` is one of `biology`, `chemistry`, `material`, `physics`, `tooluse`. You can set these
environment variables:

| Variable | Default |
|---|---|
| `LR` | 1e-5 for SDPO and SFT, 1e-6 for GRPO (GRPO is unstable at 1e-5) |
| `STEPS` | 200 |
| `TEST_FREQ` | 5 |
| `N_GPUS` | 8 |
| `MODEL` | `Qwen/Qwen3-8B` |
| `OUT_ROOT` | `outputs/sdpo_sci` |
| `RUN_NAME` | `<method>-<dataset>-lr<LR>-s<SEED>` |
| `WANDB_PROJECT` | `sdpo-sci` |
| `WANDB_DIR` | `$OUT_ROOT` |

Any extra arguments are passed to Hydra. `paper_grid.sh` runs its cells one after another. With
`DRY_RUN=1` it prints one command per run instead, which you can hand to a scheduler.

**Shared settings:**
- Batch of 32 questions, constant learning rate with no warmup.
- AdamW, weight decay 0.01, gradient clipping at 1.0.
- Maximum prompt length 2048, maximum response length 8192.
- Validation every 5 steps on the test split: 16 samples per prompt at temperature 0.6 and top-p
  0.95, scored by `verl/utils/reward_score/feedback`.

**RL** (`sdpo.yaml` / `baseline_grpo.yaml`):
- 8 rollouts per question at temperature 1.0, PPO mini-batch 32, on-policy.
- Token-level rollout importance sampling, clipped at 2.0.
- Clip-higher 0.28, no KL penalty.
- Validation also runs before the first step.

**SDPO** adds:
- Top-100 logit distillation with a tail bucket.
- Jensen–Shannon divergence (alpha 0.5).
- EMA teacher with rate 0.05.
- `max_model_len` 18944 to fit the reprompted teacher context (GRPO uses 10240).

**SFT** (`verl/trainer/fsdp_sft_trainer.py`):
- Model strategy `fsdp`, micro-batch 1, `max_length` 8192.
- Seeded with `trainer.seed`.
- Training uses verl's FSDP SFT trainer. Only validation generates text, using Hugging Face
  `generate` on an unsharded copy of the weights (`trainer.val_generation`).

RL runs seed the data order with `data.seed`. The vLLM sampling seed isn't pinned, so seeds vary
both data order and sampling. No checkpoints are written (`save_freq=-1`).

## Analysis

Both trainers log `val-core/<source>/acc/mean@16` to wandb and to the console.
`analysis/sdpo_sci/val_acc.py` reads it from the local `.wandb` files, which are written in both
online and offline mode:

```bash
python analysis/sdpo_sci/val_acc.py table  --wandb-dir outputs/sdpo_sci/wandb
python analysis/sdpo_sci/val_acc.py figure --wandb-dir outputs/sdpo_sci/wandb --out figure4.pdf
```

**Figure 4** shows seed mean ± std over the validation steps that every seed logged.

**Table 7** picks, for each dataset and method, the step with the highest seed-mean accuracy and
reports the seed std at that step. This is a best checkpoint selected on the same validation split,
so it is optimistic for every method. `--lr method=value` selects other arms. On the paper runs the
script reproduces Table 7. The paper rounds .x5 ties half-up, for example GRPO physics 71.35 → 71.4.

## Caveats

- **ToolAlpaca scoring is exact match and brittle.** Actions and their JSON inputs must match
  exactly, so an `int` vs. `str` value or an extra key scores 0. The non-greedy `Action Input`
  regex also truncates nested JSON. The rendered ground-truth targets themselves score 97.1% on
  validation (66/68) and 95.0% on train, so that is the ceiling. RL optimizes against this scorer
  directly; SFT does not.
- **SFT has no step-0 validation point.** Its first point is at step 5. The RL curves start at
  step 0 (the base model).
- **SFT validation uses Hugging Face `generate`, and RL validation uses vLLM.** The decoding
  parameters are the same.
- **SFT and SDPO peak mid-run and then decline.** GRPO at 1e-6 is often still improving at step
  200, so its Table 7 entry is a lower bound for longer training.
