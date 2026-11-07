# Antibody co-folding confidence calibration

Code, trained models and analysis for "Graph based Confidence Recalibration for Antibody Co-folding Models" (manuscript in preparation).

## Overview

We benchmark the interface confidence score (ipTM) of four co-folding models (Boltz-2, OpenFold3, Protenix, Chai-1) on 304 antibody-antigen complexes, analyse why it is miscalibrated, and train PAE-GAT, a graph attention network that uses the inter-chain predicted aligned error (PAE) matrix to recalibrate it. Chai-1 does not output PAE, so PAE-GAT uses the other three models.

## Dataset

304 complexes from SAbDab (X-ray, resolution <= 2.5 A, protein antigens, deposited after 1 June 2023): 171 conventional VH+VL antibodies and 133 VHH nanobodies. Five seeds per complex per model.

## Main results

- ipTM is miscalibrated in all four models (ECE 0.207 to 0.320; false positive rate at ipTM >= 0.7 of 15.8% to 44.9%).
- PAE-GAT, trained on two models and tested on the third, reduces ECE by 46.6% on average (7.9% to 85.2% depending on the held-out model).
- Training only on structures deposited before 2025 and testing on 2025 onwards reduces ECE by 47.6% on average.
- PAE-GAT improves calibration. It does not improve rank correlation with DockQ_HA.

## Repository structure

- `scripts/`: data download, preparation and analysis pipeline (01 to 24)
- `notebooks/`: PAE-GAT training and evaluation
  - `PAE_GAT_V1_leave_one_model_out.ipynb`: leave-one-model-out experiment
  - `PAE_GAT_Temporal_Holdout.ipynb`: temporal holdout experiment
- `weights/`: trained PAE-GAT models (`leave_one_model_out/` and `temporal_holdout/`, one per round)
- `data/`: benchmark table and analysis outputs
- `figures/`: figures used in the manuscript

## Using the trained weights

The model class `PAEGAT` is defined in the notebooks. The first cell of each notebook lists the packages to install.

```python
import torch
model = PAEGAT()
model.load_state_dict(torch.load("weights/leave_one_model_out/paegat_v1_round1.pt"))
model.eval()
```

Round 1 was trained on OpenFold3 and Protenix and tested on Boltz-2. Round 2 was trained on Boltz-2 and Protenix and tested on OpenFold3. Round 3 was trained on Boltz-2 and OpenFold3 and tested on Protenix.

## Not included in this repository

The PAE graph datasets are not included because of their size.

## Citation

Manuscript in preparation.
