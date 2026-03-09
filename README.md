# CHRONOS

**Causal-Temporal Representation Learning for Comorbidity Progression Risk in Longitudinal Primary-Care Data**
![CHRONOS architecture](https://github.com/FLaTNNBio/CHRONOS/blob/main/fig/CHRONOS.png)
## Overview

CHRONOS is a research framework for confounding-adjusted disease progression discovery from longitudinal observational primary-care records. The framework is designed to move from raw temporal co-occurrence patterns toward more interpretable progression-risk relations by combining causal design and representation learning.

More specifically, CHRONOS integrates edge-level target trial emulation, temporal sequence encoding, mutual-information-based balancing, effect-guided Siamese contrastive regularization, and cross-fitted doubly robust estimation.

## Repository Scope

This repository provides the codebase used to preprocess longitudinal records, construct edge-specific emulated trial cohorts, train the CHRONOS model, estimate edge-level effects, and generate analysis outputs for progression-network discovery.

The repository is intended to support methodological transparency and code-level reproducibility of the CHRONOS framework.


## Data Availability

The real-world data used in this project are **not publicly available**.

Because the study relies on sensitive longitudinal primary-care records subject to privacy, governance, and ethical restrictions, **raw patient-level data cannot be shared, redistributed, or released in any form through this repository**. This includes raw tables, processed patient-level datasets, and derived intermediate files that could still expose sensitive clinical information.

For this reason, this repository contains the codebase and analysis workflow, but not the underlying real-world dataset.

## Example Data

The folder `chronos_example_data/` contains **synthetic example files** created only to illustrate the expected input structure and file format used by the pipeline.

These files:

- do **not** contain real patient data
- do **not** correspond to the original study cohort
- are provided as toy examples for documentation and repository usability

Field names may reflect the structure of the original administrative schema, but all records in this folder are artificial.

## Installation

Create a Python environment and install the required dependencies:

```bash
pip install -r requirements.txt
Running the Pipeline
A typical workflow consists of three stages.
1. Build the processed dataset
python scripts/build_dataset.py --config configs/default.yaml
2. Run an edge-specific DX->DX trial
python -m src.chronos.experiments.run_dx_dx --exposure ENDO --outcome CIRC
3. Aggregate edge-level results
python scripts/summarize_results.py
```
