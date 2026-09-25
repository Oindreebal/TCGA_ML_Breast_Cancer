```r
# ============================================================
# TCGA-BRCA DATA DOWNLOAD
# ============================================================
#
# Project:
#   TCGA Breast Cancer PAM50 Classification
#
# Purpose:
#   Download and save:
#     1. TCGA-BRCA RNA-seq STAR-Counts
#     2. TCGA-BRCA sample/clinical metadata
#     3. TCGA-BRCA PAM50 subtype calls
#
# Output directory:
#   /home/oindree-bal/TCGA_ML/01_data
#
# IMPORTANT:
#   This script ONLY downloads and stores raw data.
#   No filtering or preprocessing is performed here.
#
# ============================================================


# ------------------------------------------------------------
# 0. PACKAGES
# ------------------------------------------------------------

if (!requireNamespace("BiocManager", quietly = TRUE)) {
    install.packages("BiocManager")
}

required_bioc <- c(
    "TCGAbiolinks",
    "SummarizedExperiment"
)

for (pkg in required_bioc) {
    if (!requireNamespace(pkg, quietly = TRUE)) {
        BiocManager::install(pkg, ask = FALSE, update = FALSE)
    }
}

required_cran <- c(
    "dplyr",
    "stringr",
    "readr"
)

for (pkg in required_cran) {
    if (!requireNamespace(pkg, quietly = TRUE)) {
        install.packages(pkg)
    }
}


library(TCGAbiolinks)
library(SummarizedExperiment)
library(dplyr)
library(stringr)
library(readr)


# ------------------------------------------------------------
# 1. PROJECT DIRECTORIES
# ------------------------------------------------------------

PROJECT <- "/home/oindree-bal/TCGA_ML"

DATA_DIR <- file.path(PROJECT, "01_data")

EXPR_DIR <- file.path(DATA_DIR, "TCGA_expression")
META_DIR <- file.path(DATA_DIR, "TCGA_metadata")
PAM50_DIR <- file.path(DATA_DIR, "TCGA_PAM50")

RAW_EXPR_DIR <- file.path(EXPR_DIR, "raw")

dir.create(DATA_DIR, recursive = TRUE, showWarnings = FALSE)
dir.create(EXPR_DIR, recursive = TRUE, showWarnings = FALSE)
dir.create(META_DIR, recursive = TRUE, showWarnings = FALSE)
dir.create(PAM50_DIR, recursive = TRUE, showWarnings = FALSE)
dir.create(RAW_EXPR_DIR, recursive = TRUE, showWarnings = FALSE)


cat("\n")
cat("============================================================\n")
cat("TCGA-BRCA DATA DOWNLOAD\n")
cat("============================================================\n")
cat("Project directory:", PROJECT, "\n")
cat("============================================================\n\n")


# ------------------------------------------------------------
# 2. QUERY TCGA-BRCA RNA-SEQ DATA
# ------------------------------------------------------------

cat("STEP 1: Querying TCGA-BRCA RNA-seq data...\n\n")

query <- GDCquery(
    project = "TCGA-BRCA",
    data.category = "Transcriptome Profiling",
    data.type = "Gene Expression Quantification",
    workflow.type = "STAR - Counts",
    experimental.strategy = "RNA-Seq"
)

cat("Query created successfully.\n")
cat("Number of files in query:",
    nrow(getResults(query)),
    "\n\n")


# ------------------------------------------------------------
# 3. DOWNLOAD EXPRESSION DATA
# ------------------------------------------------------------

cat("STEP 2: Downloading expression data...\n")
cat("This may take some time depending on your connection.\n\n")

GDCdownload(
    query,
    directory = RAW_EXPR_DIR
)

cat("\nExpression download completed.\n\n")


# ------------------------------------------------------------
# 4. PREPARE SUMMARIZED EXPERIMENT
# ------------------------------------------------------------

cat("STEP 3: Preparing SummarizedExperiment...\n")
cat("This can require substantial RAM and disk space.\n\n")

expr_se <- GDCprepare(
    query,
    directory = RAW_EXPR_DIR
)

cat("\nSummarizedExperiment created.\n\n")


# ------------------------------------------------------------
# 5. INSPECT THE OBJECT
# ------------------------------------------------------------

cat("============================================================\n")
cat("EXPRESSION OBJECT SUMMARY\n")
cat("============================================================\n")

cat("Class:\n")
print(class(expr_se))

cat("\nDimensions:\n")
print(dim(expr_se))

cat("\nAssays available:\n")
print(assayNames(expr_se))

cat("\nNumber of genes:", nrow(expr_se), "\n")
cat("Number of samples:", ncol(expr_se), "\n\n")


# ------------------------------------------------------------
# 6. SAVE RAW SUMMARIZED EXPERIMENT
# ------------------------------------------------------------

raw_rds <- file.path(
    EXPR_DIR,
    "TCGA_BRCA_SE_raw.rds"
)

saveRDS(
    expr_se,
    raw_rds
)

cat("Raw SummarizedExperiment saved to:\n")
cat(raw_rds, "\n\n")


# ------------------------------------------------------------
# 7. EXTRACT SAMPLE METADATA
# ------------------------------------------------------------

cat("STEP 4: Extracting sample metadata...\n\n")

meta <- as.data.frame(
    colData(expr_se)
)

cat("Metadata dimensions:",
    nrow(meta),
    "samples x",
    ncol(meta),
    "variables\n\n")


# ------------------------------------------------------------
# 8. SAVE RAW METADATA
# ------------------------------------------------------------

meta_rds <- file.path(
    META_DIR,
    "TCGA_metadata_raw.rds"
)

meta_csv <- file.path(
    META_DIR,
    "TCGA_metadata_raw.csv"
)

saveRDS(
    meta,
    meta_rds
)

write.csv(
    meta,
    meta_csv,
    row.names = FALSE
)

cat("Metadata saved:\n")
cat(meta_rds, "\n")
cat(meta_csv, "\n\n")


# ------------------------------------------------------------
# 9. CREATE BASIC BARCODE TABLE
# ------------------------------------------------------------

cat("STEP 5: Creating barcode/sample ID table...\n\n")

# Find barcode column
barcode_candidates <- c(
    "barcode",
    "sample",
    "sample_id",
    "submitter_id"
)

barcode_col <- barcode_candidates[
    barcode_candidates %in% colnames(meta)
][1]

if (is.na(barcode_col)) {

    stop(
        paste(
            "Could not automatically identify the barcode column.",
            "Available metadata columns are:",
            paste(colnames(meta), collapse = ", ")
        )
    )

}

cat("Barcode column detected:", barcode_col, "\n")


meta$full_barcode <- as.character(
    meta[[barcode_col]]
)


# ------------------------------------------------------------
# 10. CREATE PATIENT IDs
# ------------------------------------------------------------

meta$patient_id <- substr(
    meta$full_barcode,
    1,
    12
)


# ------------------------------------------------------------
# 11. PARSE SAMPLE TYPE CODE
# ------------------------------------------------------------

meta$sample_type_code <- str_sub(
    meta$full_barcode,
    14,
    15
)


meta$sample_type_label <- case_when(

    meta$sample_type_code == "01" ~ "Primary Tumor",

    meta$sample_type_code == "02" ~ "Recurrent Tumor",

    meta$sample_type_code == "06" ~ "Metastatic",

    meta$sample_type_code == "11" ~ "Solid Tissue Normal",

    TRUE ~ paste0(
        "Other (",
        meta$sample_type_code,
        ")"
    )
)


id_table <- meta %>%
    select(
        full_barcode,
        patient_id,
        sample_type_code,
        sample_type_label
    ) %>%
    distinct()


id_table_file <- file.path(
    META_DIR,
    "barcode_id_table_raw.csv"
)

write.csv(
    id_table,
    id_table_file,
    row.names = FALSE
)

cat("Barcode table saved to:\n")
cat(id_table_file, "\n\n")


# ------------------------------------------------------------
# 12. DOWNLOAD PAM50 SUBTYPE INFORMATION
# ------------------------------------------------------------

cat("STEP 6: Obtaining TCGA PAM50 subtype calls...\n\n")

subtypes <- PanCancerAtlas_subtypes()


cat("PAM50/subtype object dimensions:\n")
print(dim(subtypes))

cat("\nColumns:\n")
print(colnames(subtypes))


# ------------------------------------------------------------
# 13. SELECT BRCA
# ------------------------------------------------------------

if (!"cancer.type" %in% colnames(subtypes)) {

    stop(
        "The downloaded PanCancerAtlas_subtypes() object does not contain ",
        "'cancer.type'. Inspect the object before continuing."
    )

}

brca_subtypes <- subtypes %>%
    filter(cancer.type == "BRCA")


cat("\nNumber of BRCA subtype records:",
    nrow(brca_subtypes),
    "\n\n")


# ------------------------------------------------------------
# 14. SAVE RAW PAM50 DATA
# ------------------------------------------------------------

pam50_raw_file <- file.path(
    PAM50_DIR,
    "PAM50_raw.csv"
)

write.csv(
    brca_subtypes,
    pam50_raw_file,
    row.names = FALSE
)

cat("Raw BRCA PAM50 data saved to:\n")
cat(pam50_raw_file, "\n\n")


# ------------------------------------------------------------
# 15. CREATE DOWNLOAD MANIFEST
# ------------------------------------------------------------

manifest <- data.frame(

    item = c(
        "Project",
        "Expression workflow",
        "Expression data type",
        "Number of genes",
        "Number of samples",
        "Metadata variables",
        "PAM50 records"
    ),

    value = c(
        "TCGA-BRCA",
        "STAR - Counts",
        "Gene Expression Quantification",
        nrow(expr_se),
        ncol(expr_se),
        ncol(meta),
        nrow(brca_subtypes)
    )

)


manifest_file <- file.path(
    DATA_DIR,
    "download_manifest.csv"
)

write.csv(
    manifest,
    manifest_file,
    row.names = FALSE
)


# ------------------------------------------------------------
# 16. SESSION INFORMATION
# ------------------------------------------------------------

session_file <- file.path(
    DATA_DIR,
    "download_sessionInfo.txt"
)

sink(session_file)

cat("TCGA-BRCA data download\n")
cat("Date:", as.character(Sys.time()), "\n\n")

cat("Project:", "TCGA-BRCA", "\n")
cat("Workflow:", "STAR - Counts", "\n\n")

sessionInfo()

sink()


# ------------------------------------------------------------
# 17. FINAL MESSAGE
# ------------------------------------------------------------

cat("\n")
cat("============================================================\n")
cat("DOWNLOAD COMPLETE\n")
cat("============================================================\n")

cat("\nFiles created:\n\n")

cat("Expression:\n")
cat("  ", raw_rds, "\n\n")

cat("Metadata:\n")
cat("  ", meta_rds, "\n")
cat("  ", meta_csv, "\n\n")

cat("PAM50:\n")
cat("  ", pam50_raw_file, "\n\n")

cat("QC/ID information:\n")
cat("  ", id_table_file, "\n\n")

cat("Manifest:\n")
cat("  ", manifest_file, "\n\n")

cat("============================================================\n")
```

