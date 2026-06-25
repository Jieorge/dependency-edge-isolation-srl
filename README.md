# SRL Dependency Edge Isolation Experiments

This repository contains the experiment code for the MA thesis project
**Quantifying the Contribution of Dependency Relation Types to Semantic Role Labeling via Edge Isolation Experiments**.

The code trains and evaluates a BiLSTM + syntactic GCN semantic role labeling (SRL) model on the English CoNLL-2009 dataset. It supports three experimental settings:

- **Baseline**: train with the full predicted dependency graph.
- **Ablation**: train with one dependency-relation group masked out.
- **Isolation**: train with only one dependency-relation group retained; all other dependency arcs are redirected to self-loops.

The main thesis results use the isolation setting, where each relation group's score is compared against a zero-edge self-loop baseline.

## Minimal Files to Upload

For GitHub review or thesis replication, the minimal code artifact is:

```text
README.md
requirements.txt
shared.py
train.py
evaluate.py
```

These files are sufficient to inspect the model, data processing, training setup, and evaluation logic.

The original directory also contains `.slurm` and `.sh` files used to submit jobs on the ALICE HPC cluster. Those scripts are not required for understanding or reproducing the Python experiment logic, and they may contain cluster-specific paths, email addresses, module names, or virtual-environment locations. They should only be uploaded after being converted into generic templates.

## Expected Data Layout

By default, the scripts expect the following local layout:

```text
data/
  conll2009/
    CoNLL2009-ST-English-train.txt
    CoNLL2009-ST-English-development.txt
  glove.2024.wikigiga.100d.txt
```

The CoNLL-2009 data and GloVe vectors are not included in this repository. If your files are stored elsewhere, pass explicit paths with `--train_file`, `--dev_file`, and `--glove_file`.

## Installation

Create an environment and install the required packages:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows PowerShell, activate the environment with:

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

A CUDA-enabled PyTorch installation is recommended for full training runs.

## Dependency Relation Groups

The experiments use seven functionally defined dependency-relation groups:

| Group | Labels |
| --- | --- |
| `core_args` | `SBJ`, `OBJ`, `OPRD`, `PRD`, `PUT`, `DTV`, `LGS` |
| `clausal` | `VC`, `SUB`, `EXTR`, `IM` |
| `adjuncts` | `ADV`, `TMP`, `LOC`, `DIR`, `MNR`, `EXT`, `BNF`, `PRP`, `PRD-PRP`, `PRD-TMP`, `LOC-OPRD`, `LOC-PRD`, `MNR-TMP` |
| `noun_internal` | `NMOD`, `AMOD`, `APPO`, `HMOD`, `NAME`, `TITLE`, `POSTHON`, `SUFFIX`, `PMOD` |
| `coordination` | `COORD`, `CONJ` |
| `functional` | `DEP`, `P`, `HYPH`, `PRT`, `PRN`, `ADV-GAP`, `VOC` |
| `gap` | `GAP-SBJ`, `GAP-OBJ`, `GAP-LOC`, `GAP-TMP`, `GAP-NMOD`, `GAP-OPRD`, `GAP-PMOD`, `GAP-PRD`, `GAP-VC`, `GAP-LGS`, `DEP-GAP`, `DIR-GAP`, `EXT-GAP` |

The special `no_edge` condition is used only for isolation experiments. It replaces all dependency arcs with self-loops and serves as the zero-edge reference model.

## Training

Train the full-graph baseline:

```bash
python train.py --mode baseline
```

Train an ablation model, for example with core argument relations masked out:

```bash
python train.py --mode ablation --group core_args
```

Train the zero-edge isolation baseline:

```bash
python train.py --mode isolation --group no_edge
```

Train an isolation model, for example retaining only core argument relations:

```bash
python train.py --mode isolation --group core_args
```

Useful options:

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

Checkpoints are written to:

```text
checkpoints/
checkpoints_ablation/
checkpoints_isolation/
```

## Evaluation

Evaluate one ablation model against the full-graph baseline:

```bash
python evaluate.py --mode ablation --group core_args
```

Evaluate one isolation model against the zero-edge baseline:

```bash
python evaluate.py --mode isolation --group core_args
```

After all groups have been evaluated, generate summary plots:

```bash
python evaluate.py --mode ablation --group all
python evaluate.py --mode isolation --group all
```

Outputs are written to `results_ablation/` or `results_isolation/`. The evaluation script produces:

- overall F1 summaries by dependency group,
- per-role delta F1 tables,
- sentence-level CSV files with F1, MDD, MNDRD, and delta F1,
- MDD/MNDRD scatter plots,
- summary bar plots and role heatmaps.

## Reproducing the Main Isolation Workflow

A minimal sequential reproduction of the isolation experiments is:

```bash
python train.py --mode isolation --group no_edge

for group in core_args clausal adjuncts noun_internal coordination functional gap; do
  python train.py --mode isolation --group "$group"
  python evaluate.py --mode isolation --group "$group"
done

python evaluate.py --mode isolation --group all
```

The original experiments were run as separate HPC jobs, but the Python commands above express the same dependency structure.

## Notes

- The scripts use predicted dependency parses from the CoNLL-2009 English data.
- Non-retained dependency arcs are redirected to self-loops rather than removed from the tensor representation.
- The zero-edge baseline still passes through the GCN self-loop transformation, but contains no cross-token dependency message passing.
- Random seeds are set through `--seed`, but exact results may still vary slightly across hardware and PyTorch/CUDA versions.
