# K-B: Continuous-Domain Positional Encoding Transformers

A research collaboration between **Keenan** and **Brian** to design, analyze, and
evaluate transformer architectures that use **continuous-domain positional
encodings** — positional representations defined as continuous functions over
some underlying domain (time, space, graph coordinates, physical fields, etc.)
rather than discrete integer token indices.

Our aim is to develop a single, principled formulation that is broadly applicable
across modalities and tasks, and to characterize where it helps, where it hurts,
and why.

---

## Motivation

Standard transformer positional encodings (sinusoidal, learned absolute, RoPE,
ALiBi, etc.) treat position as a discrete token index. Many problems, however,
have an intrinsic continuous geometry:

- **Time series & signals** with irregular or multi-scale sampling
- **Physical / scientific ML**: PDE solutions, particle systems, fields
- **Spatial data**: point clouds, geospatial, molecules
- **Multimodal alignment** where tokens from different streams live on a shared
  continuous axis (e.g. time)
- **Length generalization** to positions never seen during training

A positional encoding that is *natively* a function of a continuous coordinate
should compose more naturally with these settings, support principled
extrapolation, and unify several existing tricks (interpolation, NTK scaling,
Fourier features, etc.) under one formulation.

---

## Research Questions

1. What is the right functional family for continuous positional encodings
   (Fourier features, learned spectral bases, neural fields, kernel-induced
   embeddings, ...) and how do they relate to RoPE / sinusoidal / ALiBi as
   special cases?
2. How does the continuous formulation interact with attention — additive bias,
   rotary multiplication, kernelized attention, or something new?
3. Under what conditions do we get provable / empirical **length and resolution
   generalization**?
4. How does it transfer across domains (1D time, 2D/3D space, graphs, manifolds)?
5. What is the compute / memory / quality tradeoff vs. strong discrete baselines?

---

## Literature to Review

A living reading list. Add entries as `- [ ] Author (Year). *Title*. venue. — short note on relevance.`
Promote to `- [x]` once read and summarized in `notes/`.

### Positional encodings in transformers
- [ ] Vaswani et al. (2017). *Attention Is All You Need*. — sinusoidal baseline.
- [ ] Shaw et al. (2018). *Self-Attention with Relative Position Representations*.
- [ ] Su et al. (2021). *RoFormer: Enhanced Transformer with Rotary Position Embedding (RoPE)*.
- [ ] Press et al. (2022). *Train Short, Test Long: Attention with Linear Biases (ALiBi)*.
- [ ] Chen et al. (2023). *Extending Context Window of LLMs via Positional Interpolation*.
- [ ] bloc97 / Peng et al. (2023). *NTK-Aware Scaled RoPE / YaRN*.
- [ ] Sun et al. (2023). *A Length-Extrapolatable Transformer (xPos)*.
- [ ] Kazemnejad et al. (2023). *The Impact of Positional Encoding on Length Generalization in Transformers*.

### Continuous / functional representations relevant to PEs
- [ ] Tancik et al. (2020). *Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains*.
- [ ] Sitzmann et al. (2020). *Implicit Neural Representations with Periodic Activation Functions (SIREN)*.
- [ ] Mildenhall et al. (2020). *NeRF*. — positional encoding of continuous coordinates.
- [ ] Rahimi & Recht (2007). *Random Features for Large-Scale Kernel Machines*.

### Continuous-time / irregular sequence transformers
- [ ] Zuo et al. (2020). *Transformer Hawkes Process*.
- [ ] Shukla & Marlin (2021). *Multi-Time Attention Networks for Irregularly Sampled Time Series*.
- [ ] Zhou et al. (2021/2022). *Informer / FEDformer / Autoformer family*.

### Geometric / non-1D positional structure
- [ ] Dosovitskiy et al. (2021). *ViT*. — 2D learned positions.
- [ ] Liu et al. (2021). *Swin Transformer*. — relative position bias on a 2D grid.
- [ ] Dwivedi & Bresson (2021). *A Generalization of Transformer Networks to Graphs*.
- [ ] Kreuzer et al. (2021). *Rethinking Graph Transformers with Spectral Attention (SAN)*.
- [ ] Fuchs et al. (2020). *SE(3)-Transformers*.

### Theory & analysis
- [ ] Likhosherstov et al. (2021). *On the Expressive Power of Self-Attention Matrices*.
- [ ] Yun et al. (2020). *Are Transformers Universal Approximators of Sequence-to-Sequence Functions?*

> **Process:** when you read a paper, move it to `- [x]`, drop a 1–2 paragraph
> note in `notes/lit/<short-handle>.md`, and link it from the entry above.

---

## Repository Structure

A template — create directories lazily as we need them, but keep the top-level
shape stable so things are easy to find.

```
K-B/
├── README.md                  # this file
├── papers/                    # our own write-ups (LaTeX)
│   ├── main/                  # the primary paper
│   │   ├── main.tex
│   │   ├── sections/
│   │   ├── figures/
│   │   └── refs.bib
│   └── notes/                 # short technical notes, derivations, drafts
│       └── <topic>.tex
│
├── notes/                     # discussion in markdown — low ceremony
│   ├── meetings/              # YYYY-MM-DD-<topic>.md
│   ├── ideas/                 # half-baked proposals, open questions
│   ├── decisions/             # ADR-style: why we chose X over Y
│   └── lit/                   # one .md per paper read (summary + takeaways)
│
├── src/                       # library code (importable, tested)
│   └── kb/
│       ├── encodings/         # continuous PE implementations
│       ├── attention/         # attention variants that consume them
│       ├── models/            # full model definitions
│       ├── data/              # dataset loaders / synthetic generators
│       └── utils/
│
├── experiments/               # one directory per experiment
│   └── <YYYY-MM-DD>-<slug>/
│       ├── config.yaml
│       ├── run.py             # or notebook
│       ├── README.md          # hypothesis, setup, result, takeaway
│       └── results/           # logs, checkpoints (gitignored if large)
│
├── scripts/                   # one-off utilities, data prep, plotting
│
├── notebooks/                 # exploratory; promote useful code into src/
│
└── tests/                     # mirrors src/kb/ layout
```

### Conventions

- **`papers/`** holds anything destined for an external audience. Each paper
  gets its own subdirectory with its own `refs.bib`.
- **`notes/`** is for *us*. Don't worry about polish. Date meeting notes
  `YYYY-MM-DD-<topic>.md`. Decisions get their own short file so we can point
  to them later.
- **`src/kb/`** is the importable library. Anything used by more than one
  experiment lives here. Keep it small and well-tested.
- **`experiments/`** entries are append-only — once an experiment is run,
  don't edit its config; fork a new dated directory instead. Each one has a
  `README.md` with hypothesis → setup → result → takeaway.
- **`scripts/`** for things that don't belong in the library and aren't a full
  experiment (data downloads, figure regeneration, etc.).

---

## Working Together

- **Branches & PRs:** feature branches off `main`; PRs reviewed by the other
  collaborator before merge. Long discussion threads belong in `notes/` not in
  PR comments — link to the note from the PR.
- **Issues:** use GitHub issues for tracked work. Tag with `theory`,
  `experiment`, `infra`, `writing`.
- **Checkpoints & data:** keep out of git. Use a shared bucket / drive; record
  the path in the experiment's `README.md`.
- **Reproducibility:** every experiment pins its config, seed, and library
  version (commit hash is fine).

---

## Stack

- **Framework:** PyTorch
- **Package manager / build:** TBD (likely `uv` + `pyproject.toml`) — pin in the
  first code PR alongside a minimal `src/kb/__init__.py`.

---

## Authors

- **Keenan** — _affiliation / contact TBD (see issue tracker)_
- **Brian Bell** — brian.dux.bellorum@gmail.com
