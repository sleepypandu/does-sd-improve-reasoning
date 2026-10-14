# Does self-distillation improve reasoning?

Code for the paper's experiments on SDPO (Hübotter et al., 2026) and OPSD (Zhao et al., 2026),
together with the GRPO and SFT baselines and the gradient-alignment analysis.

## Layout

| Path | Contents |
|---|---|
| `sdpo/` | [lasgroup/SDPO@7c457fc](https://github.com/lasgroup/SDPO/tree/7c457fc) (veRL), with our changes on top |
| `opsd/` | [siyan-zhao/OPSD@7448751](https://github.com/siyan-zhao/OPSD/tree/7448751) (TRL), with our changes on top |
| `env/sdpo/`, `env/opsd/` | Pinned environments for the two stacks |
| `scripts/<experiment>/` | Single-node launchers, configured through environment variables |
| `analysis/<experiment>/` | Code that builds the paper's tables and figures from run outputs |
| `gradprobe/` | Gradient-similarity statistics over recorded gradients (pooled cosine, jackknife SE, noise scale) |
| `docs/<experiment>.md` | Setup, training, evaluation and analysis for each experiment |

The first two commits import the upstream repositories verbatim; every later commit is one of our
changes, so `git log -- sdpo` and `git log -- opsd` list exactly what we modified.

All launchers assume a single node with 8 H100 GPUs and no job scheduler; wrap them with your own
scheduler to run the grids in parallel. Every `paper_grid.sh` takes `SEEDS` and prints its commands
instead of running them with `DRY_RUN=1`. Outputs go to `outputs/<experiment>/`.

## Experiments

| Paper | Experiment | Docs |
|---|---|---|
| §4.1, Figures 1, 3–5, Tables 5–7 | SDPO vs. GRPO on LiveCodeBench v6 with disjoint train/test problems (Qwen3-8B, Qwen3-14B, Olmo-3-7B-Instruct) | [`docs/sdpo_lcb.md`](docs/sdpo_lcb.md) |
| §4.2, Table 1, Figure 6 | SDPO vs. GRPO on SciKnowEval and ToolAlpaca, each trained domain validated on all five (Qwen3-8B) | [`docs/sdpo_sci.md`](docs/sdpo_sci.md) |
| §4.3, Tables 2, 8–17 | OPSD with ref/no/shuffled $z$ and teacher thinking on/off, GRPO and SFT baselines, loss-masking ablation (Qwen3-1.7B–14B, Olmo-3-7B-Instruct) | [`docs/opsd.md`](docs/opsd.md) |
| §4.4, Appendix E.3, Tables 3, 4, 18–21 | Gradient similarity at the base model: each objective's gradient against GRPO on held-out problems, math (OPSD, GRPO, SFT) and LiveCodeBench (SDPO, GRPO) | [`docs/gradprobe.md`](docs/gradprobe.md) |

## Notes on the paper runs

- **OPSD SFT baseline.** Our runs did not include the upstream fix
  [siyan-zhao/OPSD@ae7d251](https://github.com/siyan-zhao/OPSD/commit/ae7d2519e94920c4eb6206c0c26de46d9c50abae).
  Without it, TRL trains on the dataset's `conversations` column rather than the formatted
  reference solution, so the SFT baseline is SFT on the OpenThoughts chain-of-thought traces.
  `opsd/sft_train.py` keeps this behavior.
