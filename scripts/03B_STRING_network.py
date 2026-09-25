#!/usr/bin/env python3
"""
04_STRING/01_build_STRING_network.py
====================================
STRING mapping, interaction retrieval, pathway-specific networks,
and Baseline V1 degree-centrality weights.

This preserves the STRING/network components of the original script.
"""

import sys
from pathlib import Path
import time
import pandas as pd
import requests
import networkx as nx

PROJECT = Path("/home/oindree-bal/TCGA_ML")
DATA_DIR = PROJECT / "01_data"
HALLMARK_DIR = PROJECT / "03_hallmark"
STRING_DIR = PROJECT / "04_STRING"

HALLMARK_GENESETS_DIR = HALLMARK_DIR / "hallmark_50_gene_sets"
STRING_INTERACTIONS_DIR = STRING_DIR / "interactions"
STRING_NETWORKS_DIR = STRING_DIR / "pathway_networks"
STRING_CENTRALITY_DIR = STRING_DIR / "centrality_weights"

for d in [STRING_INTERACTIONS_DIR, STRING_NETWORKS_DIR, STRING_CENTRALITY_DIR]:
    d.mkdir(parents=True, exist_ok=True)

STRING_SPECIES = 9606
STRING_SCORE_THRESHOLD = 700
STRING_BATCH_SIZE = 400
STRING_REQUEST_DELAY = 1.0
CENTRALITY_METHOD = "degree"
MIN_EDGES_FOR_CENTRALITY = 5


def string_get_ids(genes, species=STRING_SPECIES):
    print("\n" + "=" * 70 + "\n3. MAPPING GENES TO STRING IDS\n" + "=" * 70)
    genes = sorted(set(genes))
    print(f"Genes to map: {len(genes)}")

    mapping = {}
    url = "https://string-db.org/api/tsv/get_string_ids"

    for i in range(0, len(genes), STRING_BATCH_SIZE):
        batch = genes[i:i + STRING_BATCH_SIZE]
        params = {
            "identifiers": "\r".join(batch),
            "species": species,
            "limit": 1
        }
        response = requests.post(url, data=params)
        response.raise_for_status()

        lines = response.text.strip().split("\n")
        if len(lines) <= 1:
            time.sleep(STRING_REQUEST_DELAY)
            continue

        header = lines[0].split("\t")
        query_col = header.index("queryItem") if "queryItem" in header else 0
        string_col = header.index("stringId") if "stringId" in header else 1

        for line in lines[1:]:
            fields = line.split("\t")
            if len(fields) > max(query_col, string_col):
                mapping[fields[query_col]] = fields[string_col]

        print(
            f"  Batch {i // STRING_BATCH_SIZE + 1}: "
            f"{len(mapping)}/{len(genes)} mapped"
        )
        time.sleep(STRING_REQUEST_DELAY)

    print(f"Mapped: {len(mapping)}  Unmapped: {len(genes) - len(mapping)}")
    return mapping


def string_get_interactions(
    string_ids,
    species=STRING_SPECIES,
    score_threshold=STRING_SCORE_THRESHOLD
):
    print("\n" + "=" * 70 + "\n4. FETCHING STRING INTERACTIONS\n" + "=" * 70)
    string_ids = sorted(set(string_ids))
    print(
        f"STRING IDs: {len(string_ids)}  "
        f"Threshold: >{score_threshold} (0-1000 scale)"
    )

    url = "https://string-db.org/api/tsv/network"
    all_edges = []

    for i in range(0, len(string_ids), STRING_BATCH_SIZE):
        batch = string_ids[i:i + STRING_BATCH_SIZE]
        params = {
            "identifiers": "%0d".join(batch),
            "species": species,
            "required_score": score_threshold
        }

        response = requests.post(url, data=params)
        response.raise_for_status()

        lines = response.text.strip().split("\n")
        if len(lines) <= 1:
            time.sleep(STRING_REQUEST_DELAY)
            continue

        header = lines[0].split("\t")
        a_col = header.index("stringId_A") if "stringId_A" in header else 2
        b_col = header.index("stringId_B") if "stringId_B" in header else 3
        score_col = header.index("score") if "score" in header else len(header) - 1

        for line in lines[1:]:
            fields = line.split("\t")
            if len(fields) > max(a_col, b_col, score_col):
                all_edges.append({
                    "from": fields[a_col],
                    "to": fields[b_col],
                    "combined_score": float(fields[score_col])
                })

        print(
            f"  Batch {i // STRING_BATCH_SIZE + 1}: "
            f"{len(all_edges)} edges so far"
        )
        time.sleep(STRING_REQUEST_DELAY)

    edges_df = pd.DataFrame(all_edges).drop_duplicates()
    print(f"Total edges fetched: {len(edges_df)}")

    if len(edges_df) > 0:
        if edges_df["combined_score"].max() <= 1.0:
            print("WARNING: combined_score on 0-1 scale -- rescaling x1000.")
            edges_df["combined_score"] *= 1000

        edges_df = edges_df[
            edges_df["combined_score"] > score_threshold
        ].copy()

    print(f"Edges after >{score_threshold} filter: {len(edges_df)}")

    edges_df.to_csv(
        STRING_INTERACTIONS_DIR / "STRING_edges.csv",
        index=False
    )

    return edges_df


def build_pathway_networks(pathways, gene_to_string, edges_df):
    print("\n" + "=" * 70 + "\n5. BUILDING PATHWAY-SPECIFIC SUBNETWORKS\n" + "=" * 70)

    pathway_networks = {}
    summary_rows = []

    for pw_name, genes in pathways.items():
        pw_ids = {
            gene_to_string[g]
            for g in genes
            if g in gene_to_string
        }

        pw_edges = edges_df[
            edges_df["from"].isin(pw_ids) &
            edges_df["to"].isin(pw_ids)
        ]

        G = nx.Graph()
        G.add_nodes_from(pw_ids)

        for _, row in pw_edges.iterrows():
            G.add_edge(row["from"], row["to"])

        pathway_networks[pw_name] = G

        summary_rows.append({
            "pathway": pw_name,
            "n_genes_in_pathway": len(genes),
            "n_genes_mapped_to_string": len(pw_ids),
            "n_edges_in_network": G.number_of_edges()
        })

    summary_df = pd.DataFrame(summary_rows).sort_values(
        "n_edges_in_network"
    )

    summary_df.to_csv(
        STRING_NETWORKS_DIR / "network_summary.csv",
        index=False
    )

    sparse = summary_df[
        summary_df["n_edges_in_network"] < MIN_EDGES_FOR_CENTRALITY
    ]

    print(
        f"Sparse pathways (<{MIN_EDGES_FOR_CENTRALITY} edges, "
        f"fallback to unweighted): {len(sparse)}"
    )

    if len(sparse) > 0:
        print(
            sparse[
                ["pathway", "n_edges_in_network"]
            ].to_string(index=False)
        )

    return pathway_networks, summary_df


def compute_centrality(pathway_networks, summary_df, gene_to_string):
    print(
        "\n" + "=" * 70 +
        f"\n6. COMPUTING CENTRALITY (method={CENTRALITY_METHOD})\n" +
        "=" * 70
    )

    string_to_gene = {
        v: k for k, v in gene_to_string.items()
    }

    sparse_pathways = set(
        summary_df.loc[
            summary_df["n_edges_in_network"] < MIN_EDGES_FOR_CENTRALITY,
            "pathway"
        ]
    )

    rows = []

    for pw_name, G in pathway_networks.items():
        fallback = pw_name in sparse_pathways

        if fallback or G.number_of_nodes() < 2:
            centrality = {
                node: 1.0
                for node in G.nodes()
            }
        else:
            if CENTRALITY_METHOD == "degree":
                centrality = nx.degree_centrality(G)
            else:
                raise NotImplementedError(
                    f"'{CENTRALITY_METHOD}' not in Baseline V1."
                )

        for sid, val in centrality.items():
            gene = string_to_gene.get(sid)

            if gene is None:
                continue

            rows.append({
                "pathway": pw_name,
                "gene_symbol": gene,
                "string_id": sid,
                "centrality": val,
                "fallback_unweighted": fallback
            })

    centrality_df = pd.DataFrame(rows)

    centrality_df.to_csv(
        STRING_CENTRALITY_DIR / "centrality_weights.csv",
        index=False
    )

    print(
        f"Centrality rows: {len(centrality_df)}  "
        f"Fallback rows: "
        f"{centrality_df['fallback_unweighted'].sum()}"
    )

    return centrality_df


def run_string_pipeline(pathways, hallmark_df):
    all_genes = sorted(
        set(hallmark_df["gene_symbol"])
    )

    expr_file = DATA_DIR / "TCGA_expression" / "TCGA_expression_clean.csv"
    expr = pd.read_csv(expr_file, index_col=0)

    genes_in_expr = [
        g for g in all_genes
        if g in expr.index
    ]

    print(
        f"\nHallmark genes present in expression matrix: "
        f"{len(genes_in_expr)}/{len(all_genes)}"
    )

    gene_to_string = string_get_ids(genes_in_expr)

    pd.DataFrame([
        {
            "gene_symbol": g,
            "string_id": s
        }
        for g, s in gene_to_string.items()
    ]).to_csv(
        HALLMARK_GENESETS_DIR / "hallmark_gene_mapping.csv",
        index=False
    )

    edges_df = string_get_interactions(
        list(gene_to_string.values())
    )

    pathway_networks, summary_df = build_pathway_networks(
        pathways,
        gene_to_string,
        edges_df
    )

    centrality_df = compute_centrality(
        pathway_networks,
        summary_df,
        gene_to_string
    )

    return centrality_df


if __name__ == "__main__":
    from importlib.util import spec_from_file_location, module_from_spec

    hallmark_path = (
        PROJECT / "03_hallmark" / "01_load_hallmark.py"
    )

    spec = spec_from_file_location("hallmark_module", hallmark_path)
    hallmark_module = module_from_spec(spec)
    spec.loader.exec_module(hallmark_module)

    pathways, hallmark_df = hallmark_module.load_hallmark_gene_sets()
    run_string_pipeline(pathways, hallmark_df)
