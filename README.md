# Network-Guided Pathway Representation for Machine-Learning Classification of Breast Cancer Molecular Subtypes

This repository contains the full analysis pipeline for classifying PAM50 molecular subtypes of breast cancer using TCGA RNA-seq expression data, MSigDB Hallmark gene sets, and STRING protein-protein interaction network centrality to weight pathway-level features.

## Overview

The central question addressed in this project is whether weighting Hallmark pathway gene sets by their network centrality (degree, betweenness, eigenvector, or PageRank) within STRING-derived interaction networks improves PAM50 subtype classification compared to simple unweighted pathway mean expression.

Six classifiers (Logistic Regression, Elastic Net, Random Forest, XGBoost, SVM-RBF, PLS-DA) were evaluated across seven feature representations (Unweighted Hallmark, pooled Network-Weighted Hallmark, and four individual centrality schemes: V3A Degree, V3B Betweenness, V3C Eigenvector, V3D PageRank) using nested cross-validation.

## Repository Structure

```
├── config/                  # Pipeline configuration (config.yaml)
├── data/
│   ├── raw/                 # Raw PAM50 calls
│   ├── processed/           # Cleaned TCGA expression matrix + matched PAM50 labels
│   └── metadata/            # Sample metadata
├── scripts/
│   ├── 01_Data_aquisition.R         # TCGA data download
│   ├── 02_Preprocessing.R           # QC, filtering, normalization
│   ├── 03A_hallmark_pathways.py     # Hallmark gene set mapping & pathway scoring
│   ├── 03B_STRING_network.py        # STRING PPI network construction per pathway
│   ├── 04_feature_engineering.py    # Centrality-weighted feature matrices (V1–V3D)
│   └── 05_ML_SHAP.py                # Nested CV, model training, SHAP attribution
├── Results/
│   ├── 01 QC/                       # QC summaries, filtering logs, PAM50 distribution
│   ├── 02 Hallmark/                 # Hallmark gene set mapping and coverage
│   ├── 03 STRING/
│   │   ├── centrality_weights/      # Per-gene centrality scores (V1–V3 schemes)
│   │   ├── interactions/            # Filtered STRING edge lists
│   │   └── pathway_networks/        # Per-pathway networks, subtype rewiring analysis
│   ├── 04 Features/                 # Final unweighted & network-weighted feature matrices
│   ├── 05 ML/                       # Frozen final model + metadata
│   ├── 06 Validation/TCGA_nested_CV/ # Nested CV scores, confusion matrices, classification reports
│   └── 07 SHAP/                     # SHAP summary plots per subtype
└── Supplementary files & figures/   # Manuscript figures and supplementary tables
```

## Pipeline

1. **Data acquisition & preprocessing** — TCGA-BRCA RNA-seq expression and PAM50 labels are downloaded, QC-filtered (library size, missing data), and matched (`01_Data_aquisition.R`, `02_Preprocessing.R`).
2. **Hallmark pathway scoring** — The 50 MSigDB Hallmark gene sets are mapped to the expression matrix and scored as unweighted pathway means (`03A_hallmark_pathways.py`).
3. **STRING network construction** — For each Hallmark pathway, a subnetwork is built from STRING protein-protein interactions, and four centrality measures (degree, betweenness, eigenvector, PageRank) are computed per gene (`03B_STRING_network.py`).
4. **Feature engineering** — Centrality scores are used to generate network-weighted pathway activity scores, producing the pooled Network-Weighted Hallmark matrix plus four individual centrality-scheme matrices (V3A–V3D) (`04_feature_engineering.py`).
5. **Nested cross-validation & SHAP** — All feature-set × classifier combinations are evaluated using nested CV (5 outer folds); the selected model is interpreted with SHAP (`05_ML_SHAP.py`).

## Key Results

- Classification performance was strong across the board: **AUROC exceeded 0.93 for every feature representation tested**, indicating the problem is well-posed regardless of whether network weighting is applied.
- The **top-performing combinations** were tightly clustered: Network-Weighted Hallmark V3D (PageRank) + Elastic Net (macro-F1 = 0.7820 ± 0.0410) and V3D (PageRank) + Logistic Regression (macro-F1 = 0.7819 ± 0.0410, balanced accuracy = 0.7650, AUROC = 0.9756) were statistically indistinguishable from one another and from several other top representations.
- The five best combinations spanned only **0.008 macro-F1**, well within the **~0.04 outer-fold standard deviation**, so no single combination can be ranked as definitively best.
- Network weighting produced a **consistent, statistically significant benefit for tree-based ensembles** (XGBoost: +0.069 macro-F1; Random Forest: +0.044; P = 0.0625, five-fold exact Wilcoxon), but had negligible or mixed effects on linear models (Logistic, Elastic Net) and PLS-DA.
- Misclassifications concentrated at biologically continuous PAM50 boundaries (Luminal A/B switching; Normal-like tumors frequently classified as Luminal A), consistent with known subtype ambiguity.

Full details, caveats (e.g., network weighting as implicit gene filtering, STRING evidence bias, PAM50 label circularity), and SHAP-based biological interpretation are provided in the manuscript Results and Discussion sections.

## Reproducing the Analysis

```bash
# create environment
conda env create -f environment.yml
conda activate <env_name>

# run pipeline in order
Rscript scripts/01_Data_aquisition.R
Rscript scripts/02_Preprocessing.R
python scripts/03A_hallmark_pathways.py
python scripts/03B_STRING_network.py
python scripts/04_feature_engineering.py
python scripts/05_ML_SHAP.py
```

Configuration (paths, hyperparameter grids, CV folds) is set in `config/config.yaml`.

## Citation

If you use this pipeline or its results, please cite the associated manuscript (citation details to be added upon publication).


