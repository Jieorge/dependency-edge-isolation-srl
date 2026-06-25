# SRL Dependency Edge Isolation

Code for the MA thesis project **Quantifying the Contribution of Dependency Relation Types to Semantic Role Labeling via Edge Isolation Experiments**.

This repository contains a PyTorch implementation of a BiLSTM + syntactic GCN model for dependency-based semantic role labeling (SRL) on CoNLL-2009 English. The baseline architecture follows the line of work implemented in [`diegma/neural-dep-srl`](https://github.com/diegma/neural-dep-srl), which provides code for *A Simple and Accurate Syntax-Agnostic Neural Model for Dependency-based Semantic Role Labeling* and *Encoding Sentences with Graph Convolutional Networks for Semantic Role Labeling*. The main extension here is an edge-isolation setup for measuring the contribution of different dependency-relation groups.

## What to Upload

For thesis-code review, the minimal repository is:

```text
README.md
requirements.txt
shared.py
train.py
evaluate.py
```

The original `.slurm` and `.sh` files were used for ALICE HPC job submission and are not needed for inspecting or reproducing the Python experiment logic. They should only be uploaded after removing cluster-specific paths, email addresses, and environment settings.

## Data Layout

The scripts expect this layout by default:

```text
data/
  conll2009/
    CoNLL2009-ST-English-train.txt
    CoNLL2009-ST-English-development.txt
  glove.2024.wikigiga.100d.txt
```

The CoNLL-2009 data and GloVe vectors are not included. Use `--train_file`, `--dev_file`, and `--glove_file` if your files are elsewhere.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

A CUDA-enabled PyTorch installation is recommended for full training runs.

## Experiments

The code supports three settings:

- `baseline`: full predicted dependency graph.
- `ablation`: one dependency-relation group masked out.
- `isolation`: only one dependency-relation group retained; all other arcs are redirected to self-loops.

The main thesis results use isolation models compared against the `no_edge` zero-edge baseline.

Dependency groups: `core_args`, `clausal`, `adjuncts`, `noun_internal`, `coordination`, `functional`, `gap`. The special group `no_edge` is only valid for isolation.

## Training

```bash
python train.py --mode baseline
python train.py --mode isolation --group no_edge
python train.py --mode isolation --group core_args
python train.py --mode ablation --group core_args
```

Common options:

```bash
python train.py \
  --mode isolation \
  --group core_args \
  --train_file data/conll2009/CoNLL2009-ST-English-train.txt \
  --dev_file data/conll2009/CoNLL2009-ST-English-development.txt \
  --glove_file data/glove.2024.wikigiga.100d.txt \
  --epochs 20 \
  --batch_size 64
```

Checkpoints are written to `checkpoints/`, `checkpoints_ablation/`, or `checkpoints_isolation/`.

## Evaluation

```bash
python evaluate.py --mode ablation --group core_args
python evaluate.py --mode isolation --group core_args
```

After all groups have been evaluated, generate summary plots:

```bash
python evaluate.py --mode ablation --group all
python evaluate.py --mode isolation --group all
```

Outputs are written to `results_ablation/` or `results_isolation/` and include group-level F1 summaries, per-role delta F1 tables, sentence-level CSV files, MDD/MNDRD scatter plots, and summary figures.

## Isolation Workflow

A sequential version of the main isolation workflow is:

```bash
python train.py --mode isolation --group no_edge

for group in core_args clausal adjuncts noun_internal coordination functional gap; do
  python train.py --mode isolation --group "$group"
  python evaluate.py --mode isolation --group "$group"
done

python evaluate.py --mode isolation --group all
```

## Notes

- Non-retained dependency arcs are redirected to self-loops rather than removed.
- The zero-edge model has no cross-token dependency message passing, but still uses the GCN self-loop transformation.
- Exact scores may vary slightly across hardware and PyTorch/CUDA versions.
