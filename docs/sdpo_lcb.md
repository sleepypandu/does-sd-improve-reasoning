# SDPO vs. GRPO on LiveCodeBench v6

Code for the LiveCodeBench experiments (Section 4.1, Appendix E.1): SDPO and GRPO trained on
half of LiveCodeBench v6 with Qwen3-8B, Qwen3-14B and Olmo-3-7B-Instruct, four seeds each,
evaluated on held-out tests of the training problems and on held-out problems.

| path | contents |
|---|---|
| `env/sdpo/` | pinned environment (shared with the other SDPO experiments) |
| `scripts/sdpo_lcb/` | data split, training, generation dumps |
| `analysis/sdpo_lcb/` | tables, curves, Figure 1 |
| `sdpo/` | upstream SDPO (verl) with the changes listed at the end |

All commands below run from the repository root.

## Environment

```bash
bash env/sdpo/setup_env.sh               # creates env/sdpo/.venv (uv, python 3.12)
source env/sdpo/.venv/bin/activate
bash env/sdpo/verify_env.sh              # on a GPU node; prints ALL_CHECKS_PASSED
```

`requirements.txt` is a flat lockfile (torch 2.9.1, vllm 0.14.0, transformers 4.57.1,
ray 2.53.0, flash-attn 2.8.3, flashinfer 0.5.3) installed with `--no-deps`; verl is the
editable `sdpo/` package. On machines where compute nodes have no internet, download the
models beforehand and set `HF_HUB_OFFLINE=1`; `train.sh` passes the local snapshot path to
the trainer. Keep JIT caches (`TRITON_CACHE_DIR`, `TORCHINDUCTOR_CACHE_DIR`,
`VLLM_CACHE_ROOT`, ...) on node-local disk when several jobs share a file system.

## Data

```bash
bash scripts/sdpo_lcb/prepare_data.sh    # DATA_DIR=data/sdpo_lcb, SEED=0
```

This downloads LiveCodeBench v6 (`livecodebench/code_generation_lite`, 131 problems from
2025-02-01 to 2025-05-01) and, with seed 0, splits it into 65 training and 66 held-out
problems. Each training problem's test cases are split 50/50: the seen half is the training
reward, the unseen half is only evaluated. `data/sdpo_lcb/lcb_exp/` then holds

* `train.parquet`: the 65 training problems with their seen tests;
* `test.parquet`: three validation splits, distinguished by `data_source`:
  `livecodebench_train_seen` (training problems, seen tests),
  `livecodebench_train_unseen` (training problems, unseen tests) and
  `livecodebench_test_unseen` (held-out problems, all tests).

## Training

```bash
METHOD=sdpo MODEL=Qwen/Qwen3-8B SEED=0 bash scripts/sdpo_lcb/train.sh
DRY_RUN=1 bash scripts/sdpo_lcb/paper_grid.sh   # all 24 runs of the paper
```

Each run takes one node with 8 H100s. Environment variables: `METHOD` (`sdpo`|`grpo`),
`MODEL`, `SEED`, `STEPS` (200), `DATA_DIR`, `OUT_ROOT` (`outputs/sdpo_lcb`), `LOGGER`
(`['console','wandb']`; set `WANDB_PROJECT`, and `WANDB_ENTITY` if needed), `SAVE_FREQ` (25),
`MAX_CKPT_TO_KEEP` (1), and `VAL_MAX_SAMPLES` (validate on a random subset, for smoke tests).
Extra arguments are passed to hydra. A run lives in `$OUT_ROOT/lcb-<method>-<model>-seed<k>/`
(checkpoints `global_step_<N>/`, console logs `logs/train-<timestamp>.log`); re-running the
same command resumes from its latest checkpoint.

Settings (Table 22): 32 questions x 8 rollouts per step, max prompt 2048 and response 8192
tokens, thinking off, training temperature 1.0; AdamW, lr 1e-6, no warmup, weight decay 0.01,
gradient clip 1.0; rollout importance-sampling clip 2.0. Validation every 10 steps with
4 samples at temperature 0.6, top-p 0.95.

* GRPO: mini-batch 8, clip ratio high 0.28, no KL, binary (all tests pass) reward.
* SDPO: mini-batch 1, reverse KL (alpha 1.0) on the student's top-20 logits, EMA teacher
  (rate 0.01), distillation IS clip 2.0. The teacher is reprompted with a successful sibling
  rollout when one exists, and otherwise with the environment feedback of the failed attempt
  (upstream templates).
* Qwen3-14B: parameter and optimizer CPU offload; SDPO additionally computes the top-k
  reduction in chunks of 1024 tokens (`self_distillation.topk_chunk_size`).

## Generations for Figure 1

```bash
bash scripts/sdpo_lcb/dump_figure1.sh                      # DUMP_DIR=$OUT_ROOT/dumps
python scripts/sdpo_lcb/grade_dump.py --dump <dump>.json.gz   # optional, CPU only
```

`dump_figure1.sh` samples the 65 training problems 8 times each (the training rollout
settings) from Qwen3-14B (sampling seeds 0-3) and from the step-200 GRPO and SDPO
checkpoints (training seeds 0-3), one GPU per dump; FSDP checkpoints are merged to HF format
first (`verl.model_merger`). `dump_generations.py` does one model or checkpoint.
`grade_dump.py` scores a dump with the training reward function.

## Analysis

```bash
python analysis/sdpo_lcb/tables.py --logs outputs/sdpo_lcb          # Tables 4-6
python analysis/sdpo_lcb/plot_curves.py --logs outputs/sdpo_lcb     # Figures 3, 5, 6
python analysis/sdpo_lcb/plot_unique_solutions.py --dumps outputs/sdpo_lcb/dumps   # Figure 1
```

The metric is the strict score, `val-aux/livecodebench_<split>/score/mean@4`: the fraction of
problems whose sample passes every test, averaged over the 4 samples. Values are parsed from
the console logs; for a resumed run the later log wins where two overlap. Tables report
mean +- std (ddof=1) over seeds at steps 0, 80 and 200. Figure 1 counts distinct programs
among each problem's 8 samples, extracting the longest fenced code block (the grader's rule)
and comparing ASTs.

## Changes to `sdpo/`

* `data/preprocess.py`, `verl/utils/reward_score/feedback/__init__.py`: treat split-tagged
  `livecodebench_*` data sources as LiveCodeBench (strip `self` from `class Solution`
  signatures; route to the code grader). Without the first, every functional test raised a
  `TypeError` on the tagged splits.
* `verl/utils/reward_score/feedback/code.py`: bind the exception in a grader error handler.
* `verl/workers/actor/dp_actor.py` and config: optional `self_distillation.topk_chunk_size`.
