#!/usr/bin/env python3
"""
05_features/01_hallmark_features.py
====================================
Hallmark pathway feature engineering.

Produces:
    05_features/Hallmark_network_weighted.csv
    05_features/Hallmark_unweighted.csv

Baseline V1 formulas are preserved exactly:
    Score_p = sum(C_g * X_pg) / sum(C_g)
    Score_p = mean(X_pg)

No PAM50 labels are used in score calculation.
"""

from pathlib import Path
import json
import pandas as pd
import numpy as np

PROJECT = Path("/home/oindree-bal/TCGA_ML")
DATA_DIR = PROJECT / "01_data"
HALLMARK_DIR = PROJECT / "03_hallmark"
STRING_DIR = PROJECT / "04_STRING"
FEATURE_DIR = PROJECT / "05_features"

HALLMARK_GENESETS_DIR = HALLMARK_DIR / "hallmark_50_gene_sets"
STRING_CENTRALITY_DIR = STRING_DIR / "centrality_weights"

FEATURE_DIR.mkdir(parents=True, exist_ok=True)

EXPR_FILE = DATA_DIR / "TCGA_expression" / "TCGA_expression_clean.csv"
LABEL_FILE = DATA_DIR / "TCGA_PAM50" / "TCGA_PAM50.csv"

STRING_SPECIES = 9606
STRING_SCORE_THRESHOLD = 700
CENTRALITY_METHOD = "degree"
MIN_EDGES_FOR_CENTRALITY = 5
RANDOM_STATE = 42


def require_file(path):
    if not path.exists():
        raise FileNotFoundError(
            f"\nRequired file not found:\n    {path}\n"
        )


def normalize_tcga_id(value):
    value = str(value).strip()

    if value.startswith("TCGA-"):
        parts = value.split("-")

        if len(parts) >= 3:
            return "-".join(parts[:3])

    return value


def load_expression_gene_by_patient():
    print(
        "\n" + "=" * 70 +
        "\n2. LOADING EXPRESSION MATRIX\n" +
        "=" * 70
    )

    require_file(EXPR_FILE)
    require_file(LABEL_FILE)

    labels = pd.read_csv(LABEL_FILE)
    labels["patient_id"] = (
        labels["patient_id"]
        .astype(str)
        .map(normalize_tcga_id)
    )

    expr_raw = pd.read_csv(
        EXPR_FILE,
        index_col=0
    )

    print(
        "Raw expression shape:",
        expr_raw.shape
    )

    canonical_ids = set(
        labels["patient_id"]
    )

    col_overlap = len(
        canonical_ids &
        {
            normalize_tcga_id(c)
            for c in expr_raw.columns
        }
    )

    idx_overlap = len(
        canonical_ids &
        {
            normalize_tcga_id(i)
            for i in expr_raw.index
        }
    )

    if col_overlap >= idx_overlap and col_overlap > 0:
        print(
            "Orientation: genes (rows) x patients (columns)"
        )

        expr = expr_raw.copy()

        expr.columns = [
            normalize_tcga_id(c)
            for c in expr.columns
        ]

    elif idx_overlap > 0:
        print(
            "Orientation: patients (rows) x genes (columns) -- transposing"
        )

        expr = expr_raw.T.copy()

        expr.columns = [
            normalize_tcga_id(c)
            for c in expr.columns
        ]

    else:
        raise ValueError(
            "Could not match expression matrix to "
            "TCGA_PAM50.csv patient IDs."
        )

    expr = expr.apply(
        pd.to_numeric,
        errors="coerce"
    )

    if expr.columns.duplicated().sum() > 0:
        print(
            "Duplicate patient columns found -- averaging."
        )

        expr = (
            expr.T
            .groupby(level=0)
            .mean()
            .T
        )

    missing = sorted(
        canonical_ids - set(expr.columns)
    )

    if missing:
        raise ValueError(
            f"Expression matrix missing "
            f"{len(missing)} patients: {missing[:10]}"
        )

    expr = expr[
        labels["patient_id"].tolist()
    ]

    print(
        "Genes x patients (final):",
        expr.shape
    )

    print(
        "Total missing values:",
        int(expr.isna().sum().sum())
    )

    return expr, labels


def compute_pathway_scores(
    expr,
    pathways,
    centrality_df
):
    print(
        "\n" + "=" * 70 +
        "\n7. COMPUTING PATHWAY SCORES\n" +
        "=" * 70
    )

    patients = expr.columns.tolist()
    pathway_names = sorted(pathways.keys())

    weighted_scores = pd.DataFrame(
        index=patients,
        columns=pathway_names,
        dtype=float
    )

    unweighted_scores = pd.DataFrame(
        index=patients,
        columns=pathway_names,
        dtype=float
    )

    for pw_name in pathway_names:

        pw_genes_all = [
            g
            for g in pathways[pw_name]
            if g in expr.index
        ]

        unweighted_scores[pw_name] = (
            expr.loc[pw_genes_all].mean(axis=0)
            if pw_genes_all
            else np.nan
        )

        pw_weights = centrality_df[
            centrality_df["pathway"] == pw_name
        ]

        pw_weights = pw_weights[
            pw_weights["gene_symbol"].isin(expr.index)
        ]

        if len(pw_weights) == 0:
            weighted_scores[pw_name] = (
                unweighted_scores[pw_name]
            )
            continue

        genes = pw_weights[
            "gene_symbol"
        ].values

        weights = (
            pw_weights
            .set_index("gene_symbol")["centrality"]
            .reindex(genes)
            .values
        )

        X = expr.loc[genes]

        numerator = (
            X.T * weights
        ).sum(axis=1)

        denominator = weights.sum()

        weighted_scores[pw_name] = (
            numerator / denominator
            if denominator > 0
            else unweighted_scores[pw_name]
        )

    weighted_scores = (
        weighted_scores
        .reset_index()
        .rename(columns={"index": "patient_id"})
    )

    unweighted_scores = (
        unweighted_scores
        .reset_index()
        .rename(columns={"index": "patient_id"})
    )

    print(
        "Weighted matrix:",
        weighted_scores.shape,
        " Unweighted matrix:",
        unweighted_scores.shape
    )

    print(
        "Missing (weighted):",
        int(weighted_scores.isna().sum().sum()),
        " Missing (unweighted):",
        int(unweighted_scores.isna().sum().sum())
    )

    return weighted_scores, unweighted_scores


def spot_check(
    expr,
    centrality_df,
    weighted_scores,
    pathway_name=None
):
    print(
        "\n" + "=" * 70 +
        "\n8. SPOT-CHECK: MANUAL VERIFICATION\n" +
        "=" * 70
    )

    if pathway_name is None:
        pathway_name = (
            centrality_df["pathway"].iloc[0]
        )

    patient_id = (
        weighted_scores["patient_id"].iloc[0]
    )

    w = centrality_df[
        (centrality_df["pathway"] == pathway_name) &
        (centrality_df["gene_symbol"].isin(expr.index))
    ]

    if len(w) == 0:
        print(
            f"No usable genes for {pathway_name} -- skipping."
        )
        return

    manual = (
        (
            w["centrality"].values *
            expr.loc[
                w["gene_symbol"].values,
                patient_id
            ].values
        ).sum()
        /
        w["centrality"].values.sum()
    )

    pipeline = (
        weighted_scores.loc[
            weighted_scores["patient_id"] == patient_id,
            pathway_name
        ].values[0]
    )

    diff = abs(
        manual - pipeline
    )

    print(
        f"Pathway: {pathway_name}  "
        f"Patient: {patient_id}"
    )

    print(
        f"Manual: {manual:.6f}  "
        f"Pipeline: {pipeline:.6f}  "
        f"Diff: {diff:.10f}  "
        f"{'OK' if diff < 1e-8 else 'MISMATCH -- INVESTIGATE'}"
    )


def save_baseline_metadata(summary_df):
    metadata = {
        "version": "V1_baseline",
        "network_source": "STRING",
        "species": STRING_SPECIES,
        "score_threshold": STRING_SCORE_THRESHOLD,
        "centrality_method": CENTRALITY_METHOD,
        "min_edges_for_centrality": MIN_EDGES_FOR_CENTRALITY,
        "weighting_formula": "Score_p = sum(C_g * X_pg) / sum(C_g)",
        "unweighted_formula": "Score_p = mean(X_pg)",
        "n_pathways": int(summary_df.shape[0]),
        "n_pathways_using_fallback": int(
            summary_df.loc[
                summary_df["n_edges_in_network"] <
                MIN_EDGES_FOR_CENTRALITY
            ].shape[0]
        ),
        "random_state": RANDOM_STATE,
        "note": (
            "Frozen V1 baseline. Phases 3-6 "
            "(normalized weighting, alternative centrality, "
            "pathway-specific network, expression-informed weighting) "
            "are separate versioned variants -- this output must stay unmodified."
        ),
    }

    with open(
        STRING_CENTRALITY_DIR /
        "BASELINE_V1_metadata.json",
        "w"
    ) as f:
        json.dump(
            metadata,
            f,
            indent=2
        )

    print(
        "\nSaved:",
        STRING_CENTRALITY_DIR /
        "BASELINE_V1_metadata.json"
    )


def main():
    from importlib.util import spec_from_file_location, module_from_spec

    hallmark_path = (
        HALLMARK_DIR /
        "01_load_hallmark.py"
    )

    string_path = (
        STRING_DIR /
        "01_build_STRING_network.py"
    )

    hallmark_spec = spec_from_file_location(
        "hallmark_module",
        hallmark_path
    )
    hallmark_module = module_from_spec(
        hallmark_spec
    )
    hallmark_spec.loader.exec_module(
        hallmark_module
    )

    string_spec = spec_from_file_location(
        "string_module",
        string_path
    )
    string_module = module_from_spec(
        string_spec
    )
    string_spec.loader.exec_module(
        string_module
    )

    pathways, hallmark_df = (
        hallmark_module.load_hallmark_gene_sets()
    )

    expr, labels = (
        load_expression_gene_by_patient()
    )

    all_genes = sorted(
        set(hallmark_df["gene_symbol"])
    )

    genes_in_expr = [
        g
        for g in all_genes
        if g in expr.index
    ]

    print(
        f"\nHallmark genes present in expression matrix: "
        f"{len(genes_in_expr)}/{len(all_genes)}"
    )

    gene_to_string = (
        string_module.string_get_ids(
            genes_in_expr
        )
    )

    pd.DataFrame([
        {
            "gene_symbol": g,
            "string_id": s
        }
        for g, s in gene_to_string.items()
    ]).to_csv(
        HALLMARK_GENESETS_DIR /
        "hallmark_gene_mapping.csv",
        index=False
    )

    edges_df = (
        string_module.string_get_interactions(
            list(gene_to_string.values())
        )
    )

    pathway_networks, summary_df = (
        string_module.build_pathway_networks(
            pathways,
            gene_to_string,
            edges_df
        )
    )

    centrality_df = (
        string_module.compute_centrality(
            pathway_networks,
            summary_df,
            gene_to_string
        )
    )

    weighted_scores, unweighted_scores = (
        compute_pathway_scores(
            expr,
            pathways,
            centrality_df
        )
    )

    spot_check(
        expr,
        centrality_df,
        weighted_scores
    )

    weighted_out = (
        FEATURE_DIR /
        "Hallmark_network_weighted.csv"
    )

    unweighted_out = (
        FEATURE_DIR /
        "Hallmark_unweighted.csv"
    )

    weighted_scores.to_csv(
        weighted_out,
        index=False
    )

    unweighted_scores.to_csv(
        unweighted_out,
        index=False
    )

    save_baseline_metadata(
        summary_df
    )

    print(
        "\n" + "=" * 70 +
        "\nBASELINE V1 COMPLETE\n" +
        "=" * 70
    )

    print(
        "Saved:",
        weighted_out,
        "\nSaved:",
        unweighted_out
    )


if __name__ == "__main__":
    main()
