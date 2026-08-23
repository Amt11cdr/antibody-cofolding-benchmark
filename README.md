# Antibody Co-Folding Benchmark

Confidence calibration analysis of Boltz-2 and OpenFold3 at the CDR-H3/antigen interface.

## Overview

Co-folding models are deployed in drug discovery pipelines but their confidence scores (ipTM) fail at CDR-H3 — the loop that determines therapeutic antibody binding specificity. This repository contains the analysis pipeline for a systematic study of where and why confidence fails, and proposes a graph-structured solution.

## Key Findings

- ECE = 0.32 at CDR-H3/antigen interface (severe miscalibration)
- 44.6% false positive rate at ipTM ≥ 0.7
- CDR-H3 PAE < framework PAE in 74.7% of complexes (11.4x residue imbalance)
- CDR-H3 RMSD varies 6.9x more than ipTM across seeds (conformational multiplicity)
- Contact position PAE barely differs from non-contact PAE (ratio 1.06)

## Dataset

304 post-training-cutoff antibody-antigen complexes from SAbDab (X-ray, ≤2.5Å, protein antigens, deposited after June 2023). 1,520 Boltz-2 predictions (5 seeds × 304 complexes).

## Repository Structure

scripts/ — Full analysis pipeline (01-11)
data/ — Benchmark dataset and analysis results

## Author

Amritansh Tiwari — MSc AI for Medicine, University College Dublin — 2026

## Preprint

Coming soon — bioRxiv
