# Adv_ML — Reproducibility study of "Vision Transformers Need Registers"

Group project for the USI ATML course (2026). We reproduce the central claims of [Darcet et al., ICLR 2024](https://arxiv.org/abs/2309.16588).

## Structure

```
src/Kun/        # Section 2 — outlier characterisation (Figs. 2, 3, 5, Tab. 1)
src/Paolo/      # Section 3.2 — linear evaluation + Figure 8 ablation (Tabs. 2a, 2b)
src/Raffaele/   # Section 4 — feature quality + LOST object discovery
report/ATML_Report_2026/   # LaTeX report
assets/         # shared figures from the paper
```

Each member's section is self-contained: scripts produce numerical results and figures under `src/<member>/results/`, which are then referenced from `report/`.

## Contributors

- Kun Zhan — Section 2
- Paolo Deidda — Section 3.2
- Raffaele Perri — Section 4
