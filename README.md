# Does self-distillation improve reasoning?

<a href="https://arxiv.org/abs/2610.18613"><img src="https://img.shields.io/badge/arXiv-2610.18613-b31b1b?style=for-the-badge" alt="arXiv"></a>
<a href="https://jasonkena.github.io/blog/posts/does_self_distillation_improve_reasoning/"><img src="https://img.shields.io/badge/Blog-brightgreen?style=for-the-badge" alt="Blog"></a>

Code for the paper's experiments on SDPO (Hübotter et al., 2026) and OPSD (Zhao et al., 2026),
together with the GRPO and SFT baselines and the gradient-alignment analysis.

## Running the experiments

Each experiment has its own guide covering environment, data, training, evaluation and the
scripts that rebuild the paper's tables and figures. **Start from the guide for the experiment
you want to reproduce:**

| Paper | Experiment | Guide |
|---|---|---|
| §4.1, Figures 1, 3–5, Tables 5–7 | SDPO vs. GRPO on LiveCodeBench v6 with disjoint train/test problems (Qwen3-8B, Qwen3-14B, Olmo-3-7B-Instruct) | [`docs/sdpo_lcb.md`](docs/sdpo_lcb.md) |
| §4.2, Table 1, Figure 6 | SDPO vs. GRPO on SciKnowEval and ToolAlpaca, each trained domain validated on all five (Qwen3-8B) | [`docs/sdpo_sci.md`](docs/sdpo_sci.md) |
| §4.3, Tables 2, 8–17 | OPSD with ref/no/shuffled $z$ and teacher thinking on/off, GRPO and SFT baselines, loss-masking ablation (Qwen3-1.7B–14B, Olmo-3-7B-Instruct) | [`docs/opsd.md`](docs/opsd.md) |
| §4.4, Appendix E.3, Tables 3, 4, 18–21 | Gradient similarity at the base model: each objective's gradient against GRPO on held-out problems, math (OPSD, GRPO, SFT) and LiveCodeBench (SDPO, GRPO) | [`docs/gradprobe.md`](docs/gradprobe.md) |

The two training stacks have separate pinned environments: `env/sdpo/` (veRL) for the SDPO
experiments and the LiveCodeBench gradients, `env/opsd/` (TRL) for OPSD and the math gradients.
Use these rather than the install instructions in the upstream READMEs inside `sdpo/` and `opsd/`.

All launchers assume a single node with 8 H100 GPUs and no job scheduler; wrap them with your own
scheduler to run the grids in parallel. Every `paper_grid.sh` takes `SEEDS` and, with `DRY_RUN=1`,
prints its commands instead of running them. Outputs go to `outputs/<experiment>/`.

## Layout

| Path | Contents |
|---|---|
| `docs/<experiment>.md` | The guide for each experiment (see above) |
| `env/sdpo/`, `env/opsd/` | Pinned environments for the two stacks |
| `scripts/<experiment>/` | Single-node launchers, configured through environment variables |
| `analysis/<experiment>/` | Code that builds the paper's tables and figures from run outputs |
| `sdpo/` | [lasgroup/SDPO@7c457fc](https://github.com/lasgroup/SDPO/tree/7c457fc) (veRL), with our changes on top |
| `opsd/` | [siyan-zhao/OPSD@7448751](https://github.com/siyan-zhao/OPSD/tree/7448751) (TRL), with our changes on top |
| `gradprobe/` | Gradient-similarity statistics over recorded gradients (pooled cosine, jackknife SE, noise scale) |

`sdpo/` is based on [lasgroup/SDPO@7c457fc](https://github.com/lasgroup/SDPO/tree/7c457fc) and
`opsd/` on [siyan-zhao/OPSD@7448751](https://github.com/siyan-zhao/OPSD/tree/7448751); our changes
are listed in [`docs/sdpo_lcb.md`](docs/sdpo_lcb.md#changes-to-sdpo) and [`docs/opsd.md`](docs/opsd.md).

## Note on the OPSD SFT baseline

Our runs did not include the upstream fix
[siyan-zhao/OPSD@ae7d251](https://github.com/siyan-zhao/OPSD/commit/ae7d2519e94920c4eb6206c0c26de46d9c50abae).
Without it, TRL trains on the dataset's `conversations` column rather than the formatted
reference solution, so the SFT baseline is SFT on the OpenThoughts chain-of-thought traces.
`opsd/sft_train.py` keeps this behavior.

## Citation

```bibtex
@misc{adhinarta2026selfdistillation,
      title={Does self-distillation improve reasoning?},
      author={Jason Ken Adhinarta and Raina Wu and Ashutosh Sharma and Avinash Balakrishnan and Rameswar Panda and Polina Golland and Yoon Kim and Zhang-Wei Hong},
      year={2026},
      eprint={2610.18613},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2610.18613},
}
```
