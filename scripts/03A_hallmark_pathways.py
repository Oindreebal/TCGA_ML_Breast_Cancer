#!/usr/bin/env python3
"""
03_hallmark/01_load_hallmark.py
================================
Hallmark gene-set loading only.

This preserves the Hallmark component of the original Baseline V1 script.
No STRING interaction retrieval or feature scoring is performed here.
"""

from pathlib import Path
import pandas as pd

PROJECT = Path("/home/oindree-bal/TCGA_ML")
HALLMARK_GENESETS_DIR = PROJECT / "03_hallmark" / "hallmark_50_gene_sets"
LOCAL_GMT_FILE = HALLMARK_GENESETS_DIR / "h.all.v2023.2.Hs.symbols.gmt"

HALLMARK_GENESETS_DIR.mkdir(parents=True, exist_ok=True)


def parse_gmt(path):
    pathways = {}
    with open(path) as f:
        for line in f:
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 3:
                continue
            pathways[fields[0]] = [g for g in fields[2:] if g]
    return pathways


def load_hallmark_gene_sets():
    print("\n" + "=" * 70)
    print("1. LOADING HALLMARK GENE SETS")
    print("=" * 70)

    if LOCAL_GMT_FILE.exists():
        print(f"Using local .gmt file: {LOCAL_GMT_FILE}")
        pathways = parse_gmt(LOCAL_GMT_FILE)
    else:
        print("Local .gmt not found -- falling back to gseapy Enrichr mirror.")
        try:
            import gseapy as gp
        except ImportError as error:
            raise ImportError(
                f"\ngseapy not installed and no .gmt at {LOCAL_GMT_FILE}.\n"
                "Either place a Hallmark .gmt there, or `pip install gseapy`.\n"
            ) from error
        gene_sets = gp.parser.download_gmt(name="MSigDB_Hallmark_2020")
        pathways = {name: list(genes) for name, genes in gene_sets.items()}

    n_pathways = len(pathways)
    print(f"Number of Hallmark pathways: {n_pathways}")
    if n_pathways != 50:
        print(f"WARNING: expected 50 pathways, found {n_pathways}.")

    rows = [
        {"gs_name": name, "gene_symbol": g}
        for name, genes in pathways.items()
        for g in genes
    ]
    hallmark_df = pd.DataFrame(rows).drop_duplicates()
    hallmark_df.to_csv(
        HALLMARK_GENESETS_DIR / "hallmark_gene_sets.csv",
        index=False
    )

    return pathways, hallmark_df


if __name__ == "__main__":
    load_hallmark_gene_sets()
