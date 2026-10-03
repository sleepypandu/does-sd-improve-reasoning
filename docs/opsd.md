# OPSD experiments

Code for the on-policy self-distillation (OPSD) experiments of the paper, together with the GRPO
and SFT baselines they are compared against. `opsd/` is upstream
[siyan-zhao/OPSD@7448751](https://github.com/siyan-zhao/OPSD/tree/7448751) with our changes on
top (see `git log -- opsd`):

- TRL 1.9.2 / transformers 5.15 / vLLM 0.27.1 compatibility (the pinned environment below).
- `--privileged_mode {normal,none,swap}`: the teacher sees the problem's reference solution $z$,
  no solution, or another problem's solution (shuffled $z$).
- `--teacher_model_name_or_path`: a separate frozen teacher (used for Olmo-3-7B-Instruct, whose
  thinking teacher is `allenai/Olmo-3-7B-Think`).
- `accelerate_zero3.yaml` and a per-module LoRA weight sync into the colocated vLLM engine under
  ZeRO-3 (Qwen3-14B and the Olmo cross-model teacher).
- `--loss_mask {none,last_10pct,all_but_last_10pct}` for the masking ablation.
- `--enable_thinking` for GRPO training rollouts (the GRPO$_\text{nothink}$ baseline).
- The run seed sets vLLM sampling and the data order.

## Setup

```bash
bash env/opsd/setup_env.sh          # uv; creates env/opsd/.venv from the pinned uv.lock
source env/opsd/.venv/bin/activate
```

The stack is torch 2.13 + CUDA 13.0 (cu130 wheels). DeepSpeed JIT-compiles its CPU Adam op at the
first launch and needs a matching CUDA 13.0 toolkit: point `CUDA_HOME` at it. Models and datasets
come from the Hugging Face Hub (`HF_HOME` / `HF_HUB_OFFLINE` as usual); wandb logging follows the
standard `WANDB_*` variables (`WANDB_MODE=disabled` to turn it off). Everything assumes one node
with 8 H100s.

## Training

```bash
MODEL=Qwen/Qwen3-8B TEACHER_THINKING=True PRIVILEGED_MODE=normal SEED=0 bash scripts/opsd/train_opsd.sh
MODEL=Qwen/Qwen3-8B ENABLE_THINKING=False SEED=0 bash scripts/opsd/train_grpo.sh
MODEL=Qwen/Qwen3-8B SEED=0 bash scripts/opsd/train_sft.sh
```

Runs are written to `$OUT_ROOT/<run>/checkpoint-<step>` (default `OUT_ROOT=outputs/opsd`), with run
names such as `opsd-model-Qwen3-8B-teacher_thinking-True-privileged_mode-normal-seed-0`. Extra
arguments are forwarded to the training script; `MAX_STEPS`, `SAVE_STEPS` and `DRY_RUN=1` (print
the command only) are honoured by all three launchers.

The per-model settings of the paper runs live in `scripts/opsd/presets.sh`. Shared settings
(Appendix E.2):

| | OPSD | GRPO | SFT |
|---|---|---|---|
| steps / checkpoint every | 200 / 20 | 500 / 50 | 200 / 20 |
| batch | 32 prompts, 1 rollout | 4 prompts x 8 rollouts, 2 updates per batch | 32 |
| sampling | T 1.1, top-p 0.95, top-k 20, 1024 tokens | T 1.2, top-p 1.0, 16000 tokens | - |
| loss | forward KL (beta 0, lmbda 1), fixed teacher, token clip | GRPO, beta 0 | NLL |
| optimizer | lr 5e-6, grad clip 0.1 | lr 5e-6, grad clip 1.0 | lr 5e-6 |

All use LoRA r 64, alpha 128 on all projections, and a non-thinking student. Per model:

| model | GPUs x per-device x grad-accum | token clip | ZeRO | notes |
|---|---|---|---|---|
| Qwen3-1.7B | OPSD 4x4x2, GRPO 4x1x8, SFT 4x2x4 | 0.05 | 2 | |
| Qwen3-4B | OPSD 8x4x1, GRPO 8x1x4, SFT 8x2x2 | 0.05 | 2 | |
| Qwen3-8B | OPSD 8x2x2, GRPO 8x1x4, SFT 8x2x2 | 0.06 | 2 | |
| Qwen3-14B | OPSD 8x1x4, GRPO 8x1x4, SFT 8x1x4 | 0.06 | 3 | vLLM TP 8 |
| Olmo-3-7B-Instruct | OPSD 8x1x4, GRPO 8x1x4, SFT 8x2x2 | 0.06 | 2 (3 with the Think teacher) | |

Olmo-3's chat templates ignore `enable_thinking`. Its "teacher thinking off" arm is
self-distillation from the frozen Instruct weights; its "teacher thinking on" arm distills from
`allenai/Olmo-3-7B-Think` (cross-model, prompted with the Think template), and it is evaluated
with thinking off only.

## The paper grid

```bash
MODEL=Qwen/Qwen3-8B SEEDS="0 1 2 3" DRY_RUN=1 bash scripts/opsd/paper_grid.sh   # print every command
MODEL=Qwen/Qwen3-8B SEEDS="0 1 2 3" STAGE=train bash scripts/opsd/paper_grid.sh
MODEL=Qwen/Qwen3-8B SEEDS="0 1 2 3" STAGE=eval EVAL_GPUS="0 1 2 3 4 5 6 7" bash scripts/opsd/paper_grid.sh
```

For one model this runs OPSD for teacher thinking {on, off} x $z$ {ref, none, shuffled}, GRPO
with training thinking {on, off} (on only for Olmo), SFT, and for Qwen3-8B the masking ablation
(teacher thinking on, both masks x the three $z$, checkpoints every 25 steps), each for every
seed. The eval stage scores the base model and every checkpoint.

## Evaluation

```bash
MODEL=Qwen/Qwen3-8B CKPT=outputs/opsd/<run>/checkpoint-200 bash scripts/opsd/eval.sh
MODEL=Qwen/Qwen3-8B bash scripts/opsd/eval.sh            # base model
```

`opsd/eval/evaluate_math.py` samples 12 solutions per problem with vLLM (temperature 1.0, top-p
0.95 thinking / 0.8 non-thinking, up to 38912 new tokens) on AIME24, AIME25 and HMMT25, grades
the last `\boxed{}` with math_verify, and writes
`$OUT_ROOT/eval_results/eval_results_<dataset>_<model>[_<run>_checkpoint-<step>]_<mode>_temp1.0_valn12.json`.
`THINKING`, `DATASETS` and `VAL_N` select the grid; extra arguments are forwarded (e.g.
`--num_samples 2` for a quick check).

## Tables

```bash
python analysis/opsd/make_tables.py --results outputs/opsd/eval_results --out tables/opsd --cache tables/opsd/lengths.parquet
```

writes, per model, `<model>_accuracy.tex` (Tables 1 and 8-11), `<model>_length.tex` (Tables
12-16) and, when masking runs are present, `<model>_masking.tex` (Table 17), plus the selected
checkpoints. Accuracy tables report, per evaluation mode, the checkpoint with the best
seed-averaged accuracy (averaged over the three benchmarks; only checkpoints every seed has
evaluated), as mean $\pm$ std over seeds. The length tables pool the three benchmarks within a
seed and report Tok@200, Tok@best (the accuracy-selected checkpoint) and, for thinking-off
evaluation, the share of rollouts with an orphan `</think>` (a closing tag with no `<think>` of
the rollout's own). The masking table uses the same per-evaluation-mode selection. Length statistics tokenize every selected rollout with the
model's own tokenizer, so run them with `--workers` set to the available cores.

## Notes

- **SFT trains on the OpenThoughts chain of thought.** `sft_train.py` adds a formatted `text`
  column but keeps the dataset's other columns, and TRL 1.9.2 then prefers the conversational
  `conversations` column, i.e. the full OpenThoughts `<|begin_of_thought|>` trace. Our runs did not
  include upstream's fix
  [ae7d2519](https://github.com/siyan-zhao/OPSD/commit/ae7d2519e94920c4eb6206c0c26de46d9c50abae),
  so the SFT baseline is SFT on the OpenThoughts CoT; the code keeps that behaviour.
- `train_opsd.sh` disables the vLLM FlashInfer sampler by default (`VLLM_USE_FLASHINFER_SAMPLER=0`).
