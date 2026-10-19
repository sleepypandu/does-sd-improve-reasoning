# SDPO vs. GRPO on SciKnowEval and ToolAlpaca, validated across domains

Code for the science and tool-use comparison (Section 4.2, Appendix D.2, Appendix E.1, Table 1,
Figure 6, Table 23). Qwen3-8B with thinking off is trained for 200 steps by GRPO or SDPO on one of
SciKnowEval biology/chemistry/material/physics or ToolAlpaca, and validated throughout on the test
splits of all five. Each method/dataset pair runs with 4 seeds, one 8-GPU node per run. The training
code is the SDPO verl fork in `sdpo/`.

## Environment

These experiments share the pinned SDPO environment with the LiveCodeBench runs:

```bash
bash env/sdpo/setup_env.sh      # creates env/sdpo/.venv
source env/sdpo/.venv/bin/activate
```

All scripts resolve `MODEL` (default `Qwen/Qwen3-8B`) to a local snapshot directory before they
launch, because Ray workers would otherwise query the Hub. On an offline node, populate the Hugging
Face cache first and set `HF_HUB_OFFLINE=1`.

## Data

```bash
bash scripts/sdpo_sci/prepare_data.sh
```

For each dataset, this writes `{train,test}.parquet` (`data/preprocess.py`) next to the committed
JSON in `sdpo/datasets/`. It then writes `sdpo/datasets/allval/test.parquet`
(`data/build_allval.py`), the union of the five test splits: 50 biology, 210 chemistry, 94
material, 80 physics and 68 ToolAlpaca prompts, 502 in all. The output is deterministic.

- **SciKnowEval splits:** these are the committed upstream splits. `data/split_tasks.py
  --test_ratio 0.1 --seed 42` reproduces them byte for byte (`RESPLIT=1` reruns it).
- **ToolAlpaca:** this uses the upstream `datasets/tooluse` split.
- **Per-domain validation:** verl reports validation metrics per `data_source`, and the four
  SciKnowEval domains share `sciknoweval`. The validation parquet therefore relabels them
  `sciknoweval_<domain>`, which the feedback reward scores with the same multiple-choice scorer.
  Training rows keep their original labels.

## Training

```bash
METHOD=sdpo DATASET=chemistry SEED=1 bash scripts/sdpo_sci/train_rl.sh   # METHOD=sdpo|grpo
DRY_RUN=1 bash scripts/sdpo_sci/paper_grid.sh                           # 2 methods x 5 datasets x seeds 1-4
```

`DATASET` is the training domain, one of `biology`, `chemistry`, `material`, `physics`, `tooluse`.
You can set these environment variables:

| Variable | Default |
|---|---|
| `LR` | 1e-5 for SDPO, 1e-6 for GRPO (GRPO is unstable at 1e-5) |
| `STEPS` | 200 |
| `TEST_FREQ` | 10 |
| `N_GPUS` | 8 |
| `MODEL` | `Qwen/Qwen3-8B` |
| `OUT_ROOT` | `outputs/sdpo_sci` |
| `RUN_NAME` | `<method>-<dataset>-lr<LR>-s<SEED>` |
| `WANDB_PROJECT` | `sdpo-sci` |
| `WANDB_DIR` | `$OUT_ROOT` |

Any extra arguments are passed to Hydra. `paper_grid.sh` runs its cells one after another. With
`DRY_RUN=1` it prints one command per run instead, which you can hand to a scheduler.

**Shared settings** (`sdpo.yaml` / `baseline_grpo.yaml`):
- Batch of 32 questions, 8 rollouts per question at temperature 1.0, PPO mini-batch 32, on-policy.
- AdamW, weight decay 0.01, gradient clipping at 1.0, constant learning rate with no warmup.
- Maximum prompt length 2048, maximum response length 8192.
- Token-level rollout importance sampling, clipped at 2.0.
- Clip-higher 0.28, no KL penalty.
- Validation before the first step and every 10 steps on all five test splits: 16 samples per
  prompt at temperature 0.6 and top-p 0.95, scored by `verl/utils/reward_score/feedback`.

**SDPO** adds:
- Top-100 logit distillation with a tail bucket.
- Jensen–Shannon divergence (alpha 0.5).
- EMA teacher with rate 0.05.
- The teacher is conditioned on a successful sibling rollout only, without environment feedback
  (`include_environment_feedback=False`), as in upstream's generalization runs. ToolAlpaca's
  feedback would otherwise put the ground-truth call in the teacher prompt whenever no sibling
  succeeded. Questions with no successful sibling are not distilled.
- `max_model_len` 18944 to fit the reprompted teacher context (GRPO uses 10240).

Runs seed the data order with `data.seed`. The vLLM sampling seed isn't pinned, so seeds vary
both data order and sampling. No checkpoints are written (`save_freq=-1`).

## Analysis

The trainer logs `val-core/<domain>/acc/mean@16` for each evaluated domain to wandb and to the
console (`sciknoweval_<domain>` for SciKnowEval, `tooluse` for ToolAlpaca).
`analysis/sdpo_sci/val_acc.py` reads them from the local `.wandb` files, which are written in both
online and offline mode:

```bash
python analysis/sdpo_sci/val_acc.py table  --wandb-dir outputs/sdpo_sci/wandb
python analysis/sdpo_sci/val_acc.py figure --wandb-dir outputs/sdpo_sci/wandb --out figure6.pdf
```

**Table 1** reports, for each method and training domain, the seed mean ± std at step 200 on every
evaluated domain. The **Base** row is step 0, pooled over all 40 runs. **OOD change** is, per seed,
the mean over the four domains not trained on of (step-200 accuracy − Base), then mean ± std over
seeds. On the paper runs the script reproduces every cell of Table 1.

**Figure 6** is a 5×5 grid, training domain by row and evaluated domain by column, showing seed
mean ± std at every validation step. The diagonal is shaded.

Only runs that logged every domain at step 0 and step 200 are counted. `--lr method=value`
selects other learning rates and `--steps` another final step.

