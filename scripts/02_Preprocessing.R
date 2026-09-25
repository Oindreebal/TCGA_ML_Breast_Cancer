```r
# ============================================================
# TCGA-BRCA PREPROCESSING + QUALITY CONTROL
# ============================================================
#
# Input:
#   /home/oindree-bal/TCGA_ML/01_data
#
# Output:
#   /home/oindree-bal/TCGA_ML/02_preprocessing/QC
#
# Pipeline:
#
#   Raw TCGA expression
#          |
#          v
#   Primary tumour samples only
#          |
#          v
#   Match PAM50 by patient ID
#          |
#          v
#   Remove missing/invalid PAM50
#          |
#          v
#   Remove duplicate patients
#          |
#          v
#   Expression QC
#          |
#          v
#   TPM
#          |
#          v
#   log2(TPM + 1)
#          |
#          v
#   Low-expression filtering
#          |
#          v
#   Ensembl -> gene symbol
#          |
#          v
#   Final clean expression matrix
#
# ============================================================


# ------------------------------------------------------------
# 0. PACKAGES
# ------------------------------------------------------------

if (!requireNamespace("BiocManager", quietly = TRUE)) {
    install.packages("BiocManager")
}

required_bioc <- c(
    "SummarizedExperiment"
)

for (pkg in required_bioc) {

    if (!requireNamespace(pkg, quietly = TRUE)) {

        BiocManager::install(
            pkg,
            ask = FALSE,
            update = FALSE
        )

    }
}


required_cran <- c(
    "dplyr",
    "stringr",
    "tibble",
    "ggplot2"
)

for (pkg in required_cran) {

    if (!requireNamespace(pkg, quietly = TRUE)) {

        install.packages(pkg)

    }

}


library(SummarizedExperiment)
library(dplyr)
library(stringr)
library(tibble)
library(ggplot2)


# ------------------------------------------------------------
# 1. PROJECT DIRECTORIES
# ------------------------------------------------------------

PROJECT <- "/home/oindree-bal/TCGA_ML"

DATA_DIR <- file.path(
    PROJECT,
    "01_data"
)

EXPR_DIR <- file.path(
    DATA_DIR,
    "TCGA_expression"
)

META_DIR <- file.path(
    DATA_DIR,
    "TCGA_metadata"
)

PAM50_DIR <- file.path(
    DATA_DIR,
    "TCGA_PAM50"
)

PREP_DIR <- file.path(
    PROJECT,
    "02_preprocessing"
)

QC_DIR <- file.path(
    PREP_DIR,
    "QC"
)

CLEAN_EXPR_DIR <- file.path(
    PREP_DIR,
    "cleaned_expression"
)


dir.create(
    PREP_DIR,
    recursive = TRUE,
    showWarnings = FALSE
)

dir.create(
    QC_DIR,
    recursive = TRUE,
    showWarnings = FALSE
)

dir.create(
    CLEAN_EXPR_DIR,
    recursive = TRUE,
    showWarnings = FALSE
)


# ------------------------------------------------------------
# 2. LOAD RAW SUMMARIZED EXPERIMENT
# ------------------------------------------------------------

cat("\n")
cat("============================================================\n")
cat("TCGA-BRCA PREPROCESSING + QC\n")
cat("============================================================\n\n")

raw_rds <- file.path(
    EXPR_DIR,
    "TCGA_BRCA_SE_raw.rds"
)


if (!file.exists(raw_rds)) {

    stop(
        "Raw SummarizedExperiment not found:\n",
        raw_rds,
        "\nRun 01_download_TCGA_BRCA.R first."
    )

}


cat("Loading raw SummarizedExperiment...\n")

expr_se <- readRDS(
    raw_rds
)


cat("Loaded.\n")

cat("Genes:", nrow(expr_se), "\n")
cat("Samples:", ncol(expr_se), "\n\n")


# ------------------------------------------------------------
# 3. LOAD PAM50
# ------------------------------------------------------------

pam50_file <- file.path(
    PAM50_DIR,
    "PAM50_raw.csv"
)


if (!file.exists(pam50_file)) {

    stop(
        "PAM50 file not found:\n",
        pam50_file
    )

}


pam50 <- read.csv(
    pam50_file,
    stringsAsFactors = FALSE,
    check.names = FALSE
)


cat("PAM50 records:", nrow(pam50), "\n")
cat("PAM50 columns:\n")
print(colnames(pam50))
cat("\n")


# ------------------------------------------------------------
# 4. EXTRACT SAMPLE METADATA
# ------------------------------------------------------------

meta <- as.data.frame(
    colData(expr_se)
)


# Identify barcode
barcode_candidates <- c(
    "barcode",
    "sample",
    "sample_id",
    "submitter_id"
)


barcode_matches <- barcode_candidates[
    barcode_candidates %in% colnames(meta)
]


if (length(barcode_matches) == 0) {

    stop(
        "Could not find a barcode column in colData(expr_se).\n",
        "Available columns:\n",
        paste(colnames(meta), collapse = ", ")
    )

}


barcode_col <- barcode_matches[1]


cat("Using barcode column:",
    barcode_col,
    "\n\n")


meta$full_barcode <- as.character(
    meta[[barcode_col]]
)


# ------------------------------------------------------------
# 5. CREATE PATIENT ID
# ------------------------------------------------------------

meta$patient_id <- substr(
    meta$full_barcode,
    1,
    12
)


# ------------------------------------------------------------
# 6. PARSE SAMPLE TYPE
# ------------------------------------------------------------

meta$sample_type_code <- str_sub(
    meta$full_barcode,
    14,
    15
)


meta$sample_type_label <- case_when(

    meta$sample_type_code == "01" ~
        "Primary Tumor",

    meta$sample_type_code == "02" ~
        "Recurrent Tumor",

    meta$sample_type_code == "06" ~
        "Metastatic",

    meta$sample_type_code == "11" ~
        "Solid Tissue Normal",

    TRUE ~
        paste0(
            "Other (",
            meta$sample_type_code,
            ")"
        )
)


# ------------------------------------------------------------
# 7. CREATE BARCODE QC TABLE
# ------------------------------------------------------------

id_table <- meta %>%
    select(
        full_barcode,
        patient_id,
        sample_type_code,
        sample_type_label
    ) %>%
    distinct()


write.csv(
    id_table,
    file.path(
        QC_DIR,
        "barcode_id_table.csv"
    ),
    row.names = FALSE
)


# ------------------------------------------------------------
# 8. IDENTIFY PAM50 SAMPLE/PATIENT COLUMN
# ------------------------------------------------------------

pam_id_candidates <- c(
    "pan.samplesID",
    "sample",
    "sample_id",
    "patient_id"
)


pam_id_matches <- pam_id_candidates[
    pam_id_candidates %in% colnames(pam50)
]


if (length(pam_id_matches) == 0) {

    stop(
        "Could not identify the PAM50 sample/patient ID column.\n",
        "Available columns:\n",
        paste(colnames(pam50), collapse = ", ")
    )

}


pam_id_col <- pam_id_matches[1]


cat("Using PAM50 ID column:",
    pam_id_col,
    "\n")


# ------------------------------------------------------------
# 9. IDENTIFY PAM50 SUBTYPE COLUMN
# ------------------------------------------------------------

pam_label_candidates <- c(
    "Subtype_mRNA",
    "PAM50",
    "SUBTYPE",
    "subtype"
)


pam_label_matches <- pam_label_candidates[
    pam_label_candidates %in% colnames(pam50)
]


if (length(pam_label_matches) == 0) {

    stop(
        "Could not identify PAM50 subtype column.\n",
        "Available columns:\n",
        paste(colnames(pam50), collapse = ", ")
    )

}


pam_label_col <- pam_label_matches[1]


cat("Using PAM50 subtype column:",
    pam_label_col,
    "\n\n")


# ------------------------------------------------------------
# 10. HARMONISE PAM50 IDS
# ------------------------------------------------------------

pam50 <- pam50 %>%

    mutate(

        patient_id = substr(
            as.character(.data[[pam_id_col]]),
            1,
            12
        ),

        PAM50 = as.character(
            .data[[pam_label_col]]
        )

    )


# ------------------------------------------------------------
# 11. NORMALISE PAM50 LABELS
# ------------------------------------------------------------

pam50$PAM50 <- str_trim(
    pam50$PAM50
)


# Handle common BRCA-prefixed labels
pam50$PAM50 <- case_when(

    str_detect(
        pam50$PAM50,
        regex("LumA", ignore_case = TRUE)
    ) ~ "LumA",

    str_detect(
        pam50$PAM50,
        regex("LumB", ignore_case = TRUE)
    ) ~ "LumB",

    str_detect(
        pam50$PAM50,
        regex("Her2|HER2", ignore_case = TRUE)
    ) ~ "Her2",

    str_detect(
        pam50$PAM50,
        regex("Basal", ignore_case = TRUE)
    ) ~ "Basal",

    str_detect(
        pam50$PAM50,
        regex("Normal", ignore_case = TRUE)
    ) ~ "Normal",

    TRUE ~ pam50$PAM50

)


pam50 <- pam50 %>%

    select(
        patient_id,
        PAM50
    ) %>%

    distinct()


# ------------------------------------------------------------
# 12. CHECK PAM50 DUPLICATES
# ------------------------------------------------------------

pam50_duplicates <- pam50 %>%

    count(patient_id) %>%

    filter(n > 1)


write.csv(
    pam50_duplicates,
    file.path(
        QC_DIR,
        "pam50_duplicate_patients.csv"
    ),
    row.names = FALSE
)


# ------------------------------------------------------------
# 13. MATCH METADATA WITH PAM50
# ------------------------------------------------------------

merged <- id_table %>%

    left_join(
        pam50,
        by = "patient_id"
    )


# ------------------------------------------------------------
# 14. FILTERING LOG
# ------------------------------------------------------------

n0 <- nrow(merged)


# ------------------------------------------------------------
# STEP 1: PRIMARY TUMOUR ONLY
# ------------------------------------------------------------

step1 <- merged %>%

    filter(
        sample_type_label == "Primary Tumor"
    )


n1 <- nrow(step1)


cat("Raw samples:", n0, "\n")
cat("Primary tumour samples:", n1, "\n")


# ------------------------------------------------------------
# STEP 2: PAM50 PRESENT
# ------------------------------------------------------------

step2 <- step1 %>%

    filter(
        !is.na(PAM50),
        PAM50 != "",
        PAM50 != "NA"
    )


n2 <- nrow(step2)


cat("Samples with PAM50:", n2, "\n")


# ------------------------------------------------------------
# STEP 3: VALID PAM50 LABEL
# ------------------------------------------------------------

valid_labels <- c(
    "LumA",
    "LumB",
    "Her2",
    "Basal",
    "Normal"
)


step3 <- step2 %>%

    filter(
        PAM50 %in% valid_labels
    )


n3 <- nrow(step3)


cat("Samples with valid PAM50:", n3, "\n")


# ------------------------------------------------------------
# STEP 4: REMOVE DUPLICATE PATIENTS
# ------------------------------------------------------------

dup_patients <- step3 %>%

    count(patient_id) %>%

    filter(n > 1)


write.csv(
    dup_patients,
    file.path(
        QC_DIR,
        "duplicate_patients_before_dedup.csv"
    ),
    row.names = FALSE
)


# Deterministic rule:
# retain the first primary tumour sample.
#
# Because sample ordering originates from the downloaded
# GDC object, we preserve the first occurrence rather than
# randomly sampling.


step4 <- step3 %>%

    group_by(patient_id) %>%

    slice(1) %>%

    ungroup()


n4 <- nrow(step4)


cat("Unique patients:", n4, "\n\n")


# ------------------------------------------------------------
# 15. SAVE FILTER LOG
# ------------------------------------------------------------

filter_log <- data.frame(

    step = c(
        "raw",
        "primary_tumor_only",
        "pam50_present",
        "pam50_valid_label",
        "dedup_patient"
    ),

    n_samples = c(
        n0,
        n1,
        n2,
        n3,
        n4
    )

)


write.csv(
    filter_log,
    file.path(
        QC_DIR,
        "filter_log.csv"
    ),
    row.names = FALSE
)


# ------------------------------------------------------------
# 16. MATCH EXPRESSION SAMPLES
# ------------------------------------------------------------

cat("Matching expression samples...\n\n")


final_barcodes <- step4$full_barcode


expression_barcodes <- colnames(
    expr_se
)


# Make sure the requested barcodes exist
missing_expression <- setdiff(
    final_barcodes,
    expression_barcodes
)


if (length(missing_expression) > 0) {

    cat(
        "WARNING:",
        length(missing_expression),
        "metadata samples were not found in expression object.\n"
    )

    write.csv(
        data.frame(
            missing_barcode = missing_expression
        ),
        file.path(
            QC_DIR,
            "missing_expression_barcodes.csv"
        ),
        row.names = FALSE
    )

}


final_barcodes <- intersect(
    final_barcodes,
    expression_barcodes
)


expr_matched <- expr_se[
    ,
    final_barcodes
]


# Reorder metadata to EXACTLY match expression
step4 <- step4[
    match(
        colnames(expr_matched),
        step4$full_barcode
    ),
]


# ------------------------------------------------------------
# 17. MATCHING ASSERTIONS
# ------------------------------------------------------------

stopifnot(
    ncol(expr_matched) == nrow(step4)
)


stopifnot(
    all(
        colnames(expr_matched) ==
            step4$full_barcode
    )
)


stopifnot(
    !any(
        duplicated(
            step4$patient_id
        )
    )
)


cat("Expression/metadata matching successful.\n")
cat("Final matched samples:",
    ncol(expr_matched),
    "\n\n")


# ------------------------------------------------------------
# 18. SAVE MATCHED OBJECT
# ------------------------------------------------------------

saveRDS(
    expr_matched,
    file.path(
        CLEAN_EXPR_DIR,
        "TCGA_expr_matched.rds"
    )
)


write.csv(
    step4,
    file.path(
        DATA_DIR,
        "TCGA_metadata",
        "TCGA_metadata_matched.csv"
    ),
    row.names = FALSE
)


# ------------------------------------------------------------
# 19. RAW EXPRESSION QC
# ------------------------------------------------------------

cat("============================================================\n")
cat("EXPRESSION QC\n")
cat("============================================================\n")


# Determine available count assay
available_assays <- assayNames(
    expr_matched
)


cat("Available assays:\n")
print(available_assays)
cat("\n")


if ("unstranded" %in% available_assays) {

    counts <- assay(
        expr_matched,
        "unstranded"
    )

} else {

    stop(
        "The expected 'unstranded' assay was not found."
    )

}


# ------------------------------------------------------------
# 20. BASIC SHAPE
# ------------------------------------------------------------

cat(
    "Genes:",
    nrow(counts),
    "\n"
)

cat(
    "Samples:",
    ncol(counts),
    "\n"
)


# ------------------------------------------------------------
# 21. MISSING VALUES
# ------------------------------------------------------------

any_na <- any(
    is.na(counts)
)


cat(
    "Any NA in counts:",
    any_na,
    "\n"
)


# ------------------------------------------------------------
# 22. DUPLICATE ENSEMBL IDS
# ------------------------------------------------------------

duplicate_ensembl <- any(
    duplicated(
        rownames(counts)
    )
)


cat(
    "Duplicate Ensembl rownames:",
    duplicate_ensembl,
    "\n"
)


# ------------------------------------------------------------
# 23. DUPLICATE PATIENTS
# ------------------------------------------------------------

duplicate_patients <- any(
    duplicated(
        step4$patient_id
    )
)


cat(
    "Duplicate patients:",
    duplicate_patients,
    "\n"
)


# ------------------------------------------------------------
# 24. PAM50 DISTRIBUTION
# ------------------------------------------------------------

subtype_table <- table(
    step4$PAM50
)


print(subtype_table)


subtype_df <- as.data.frame(
    subtype_table
)


colnames(subtype_df) <- c(
    "PAM50",
    "Count"
)


write.csv(
    subtype_df,
    file.path(
        QC_DIR,
        "pam50_distribution.csv"
    ),
    row.names = FALSE
)


# ------------------------------------------------------------
# 25. PAM50 BARPLOT
# ------------------------------------------------------------

p_pam50 <- ggplot(
    step4,
    aes(x = PAM50)
) +
    geom_bar() +
    labs(
        title = "PAM50 subtype distribution",
        x = "PAM50 subtype",
        y = "Number of samples"
    ) +
    theme_minimal()


ggsave(
    filename = file.path(
        QC_DIR,
        "pam50_distribution.png"
    ),
    plot = p_pam50,
    width = 7,
    height = 5,
    dpi = 300
)


# ------------------------------------------------------------
# 26. LIBRARY SIZE
# ------------------------------------------------------------

cat("\nCalculating library sizes...\n")


lib_sizes <- colSums(
    counts
)


lib_size_df <- data.frame(

    full_barcode = names(
        lib_sizes
    ),

    patient_id = step4$patient_id,

    library_size = as.numeric(
        lib_sizes
    )

)


write.csv(
    lib_size_df,
    file.path(
        QC_DIR,
        "library_sizes.csv"
    ),
    row.names = FALSE
)


# ------------------------------------------------------------
# 27. LIBRARY SIZE PLOT
# ------------------------------------------------------------

p_lib <- ggplot(
    lib_size_df,
    aes(x = library_size)
) +
    geom_histogram(
        bins = 40
    ) +
    labs(
        title = "TCGA-BRCA library size distribution",
        x = "Total counts",
        y = "Number of samples"
    ) +
    theme_minimal()


ggsave(
    filename = file.path(
        QC_DIR,
        "library_size_distribution.png"
    ),
    plot = p_lib,
    width = 7,
    height = 5,
    dpi = 300
)


# ------------------------------------------------------------
# 28. MISSING-DATA SUMMARY
# ------------------------------------------------------------

missing_summary <- data.frame(

    metric = c(
        "Total samples",
        "Total genes",
        "Samples with NA counts",
        "Genes with any NA",
        "Samples missing PAM50"
    ),

    value = c(

        ncol(counts),

        nrow(counts),

        sum(
            colSums(
                is.na(counts)
            ) > 0
        ),

        sum(
            rowSums(
                is.na(counts)
            ) > 0
        ),

        sum(
            is.na(
                step4$PAM50
            )
        )

    )

)


write.csv(
    missing_summary,
    file.path(
        QC_DIR,
        "missing_data_summary.csv"
    ),
    row.names = FALSE
)


# ------------------------------------------------------------
# 29. CHECK NON-NEGATIVE COUNTS
# ------------------------------------------------------------

negative_counts <- any(
    counts < 0,
    na.rm = TRUE
)


cat(
    "Any negative counts:",
    negative_counts,
    "\n"
)


# ------------------------------------------------------------
# 30. EXTRACT TPM
# ------------------------------------------------------------

cat("\n============================================================\n")
cat("EXPRESSION PREPROCESSING\n")
cat("============================================================\n\n")


if ("tpm_unstrand" %in% assayNames(expr_matched)) {

    cat("Using GDC-provided TPM assay: tpm_unstrand\n\n")

    tpm <- assay(
        expr_matched,
        "tpm_unstrand"
    )

} else {

    stop(
        "Expected assay 'tpm_unstrand' was not found.\n",
        "Available assays:\n",
        paste(
            assayNames(expr_matched),
            collapse = ", "
        )
    )

}


# ------------------------------------------------------------
# 31. CHECK TPM
# ------------------------------------------------------------

cat(
    "TPM dimensions:",
    nrow(tpm),
    "genes x",
    ncol(tpm),
    "samples\n"
)


cat(
    "Any NA in TPM:",
    any(
        is.na(tpm)
    ),
    "\n"
)


# ------------------------------------------------------------
# 32. LOW-EXPRESSION FILTER
# ------------------------------------------------------------
#
# Rule:
#
#   TPM > 1
#   in at least 20% of final samples
#
# This is applied BEFORE gene-symbol conversion.
# ------------------------------------------------------------


expression_threshold <- 1

sample_fraction <- 0.20

minimum_samples <- ceiling(
    sample_fraction *
        ncol(tpm)
)


keep_genes <- rowSums(
    tpm > expression_threshold
) >= minimum_samples


cat("\nLow-expression filtering:\n")

cat(
    "TPM threshold:",
    expression_threshold,
    "\n"
)

cat(
    "Required sample fraction:",
    sample_fraction,
    "\n"
)

cat(
    "Required number of samples:",
    minimum_samples,
    "\n"
)

cat(
    "Genes before filtering:",
    nrow(tpm),
    "\n"
)

cat(
    "Genes retained:",
    sum(keep_genes),
    "\n"
)

cat(
    "Genes removed:",
    sum(!keep_genes),
    "\n\n"
)


gene_filter_summary <- data.frame(

    metric = c(
        "Genes before filtering",
        "Genes retained",
        "Genes removed",
        "TPM threshold",
        "Minimum sample fraction",
        "Minimum number of samples"
    ),

    value = c(
        nrow(tpm),
        sum(keep_genes),
        sum(!keep_genes),
        expression_threshold,
        sample_fraction,
        minimum_samples
    )

)


write.csv(
    gene_filter_summary,
    file.path(
        QC_DIR,
        "gene_expression_filter_summary.csv"
    ),
    row.names = FALSE
)


# ------------------------------------------------------------
# 33. APPLY GENE FILTER
# ------------------------------------------------------------

tpm_filtered <- tpm[
    keep_genes,
    ,
    drop = FALSE
]


# ------------------------------------------------------------
# 34. LOG2 TRANSFORMATION
# ------------------------------------------------------------

log_tpm_filtered <- log2(
    tpm_filtered + 1
)


# ------------------------------------------------------------
# 35. GENE ANNOTATION
# ------------------------------------------------------------

cat("Extracting gene annotation...\n\n")


gene_annotation <- as.data.frame(
    rowData(expr_matched)
)


cat("Available rowData columns:\n")
print(colnames(gene_annotation))
cat("\n")


# Find Ensembl ID column
gene_id_candidates <- c(
    "gene_id",
    "gene_id_original"
)


gene_id_matches <- gene_id_candidates[
    gene_id_candidates %in%
        colnames(gene_annotation)
]


if (length(gene_id_matches) == 0) {

    stop(
        "Could not identify gene ID column in rowData()."
    )

}


gene_id_col <- gene_id_matches[1]


# Find gene symbol/name column
gene_name_candidates <- c(
    "gene_name",
    "gene_symbol",
    "external_gene_name"
)


gene_name_matches <- gene_name_candidates[
    gene_name_candidates %in%
        colnames(gene_annotation)
]


if (length(gene_name_matches) == 0) {

    stop(
        "Could not identify gene symbol/name column in rowData()."
    )

}


gene_name_col <- gene_name_matches[1]


cat(
    "Gene ID column:",
    gene_id_col,
    "\n"
)

cat(
    "Gene symbol column:",
    gene_name_col,
    "\n\n"
)


gene_map <- gene_annotation %>%

    select(
        gene_id = all_of(gene_id_col),
        gene_name = all_of(gene_name_col)
    ) %>%

    mutate(
        gene_id = as.character(gene_id),
        gene_name = as.character(gene_name)
    ) %>%

    filter(
        gene_id %in%
            rownames(log_tpm_filtered)
    )


# ------------------------------------------------------------
# 36. ENSURE UNIQUE GENE IDS
# ------------------------------------------------------------

gene_map <- gene_map %>%

    distinct(
        gene_id,
        .keep_all = TRUE
    )


# ------------------------------------------------------------
# 37. CONVERT EXPRESSION TO DATA FRAME
# ------------------------------------------------------------

log_tpm_df <- as.data.frame(
    log_tpm_filtered
)


log_tpm_df <- log_tpm_df %>%

    rownames_to_column(
        "gene_id"
    )


# ------------------------------------------------------------
# 38. JOIN GENE SYMBOLS
# ------------------------------------------------------------

log_tpm_df <- log_tpm_df %>%

    left_join(
        gene_map,
        by = "gene_id"
    )


# ------------------------------------------------------------
# 39. REMOVE GENES WITHOUT SYMBOL
# ------------------------------------------------------------

log_tpm_df <- log_tpm_df %>%

    filter(
        !is.na(gene_name),
        gene_name != ""
    )


# ------------------------------------------------------------
# 40. COLLAPSE DUPLICATE GENE SYMBOLS
# ------------------------------------------------------------
#
# If multiple Ensembl IDs map to the same gene symbol,
# retain the transcript/gene record with the highest
# mean expression across the final cohort.
#
# The mean is calculated on the LINEAR TPM scale before
# log transformation would normally be preferable.
#
# However, because the working matrix is already log2(TPM+1),
# this script uses mean log-expression consistently with
# the supplied Week 1 workflow.
# ------------------------------------------------------------


numeric_columns <- setdiff(
    colnames(log_tpm_df),
    c(
        "gene_id",
        "gene_name"
    )
)


log_tpm_df$mean_expr <- rowMeans(
    log_tpm_df[
        ,
        numeric_columns,
        drop = FALSE
    ],
    na.rm = TRUE
)


log_tpm_df <- log_tpm_df %>%

    arrange(
        desc(mean_expr)
    ) %>%

    distinct(
        gene_name,
        .keep_all = TRUE
    ) %>%

    select(
        -gene_id,
        -mean_expr
    )


# ------------------------------------------------------------
# 41. SET GENE SYMBOLS AS ROW NAMES
# ------------------------------------------------------------

rownames(
    log_tpm_df
) <- log_tpm_df$gene_name


log_tpm_df$gene_name <- NULL


# ------------------------------------------------------------
# 42. FINAL EXPRESSION MATRIX CHECKS
# ------------------------------------------------------------

cat("\n============================================================\n")
cat("FINAL MATRIX QC\n")
cat("============================================================\n\n")


cat(
    "Final genes:",
    nrow(log_tpm_df),
    "\n"
)

cat(
    "Final samples:",
    ncol(log_tpm_df),
    "\n"
)


# No NA
final_na <- sum(
    is.na(log_tpm_df)
)


cat(
    "Final NA values:",
    final_na,
    "\n"
)


# No duplicated gene symbols
duplicate_symbols <- any(
    duplicated(
        rownames(log_tpm_df)
    )
)


cat(
    "Duplicate gene symbols:",
    duplicate_symbols,
    "\n"
)


# Expression/metadata order
sample_order_ok <- all(
    colnames(log_tpm_df) ==
        step4$full_barcode
)


cat(
    "Expression/metadata sample order correct:",
    sample_order_ok,
    "\n"
)


# Patient duplication
patient_duplicates_final <- any(
    duplicated(
        step4$patient_id
    )
)


cat(
    "Duplicate patients:",
    patient_duplicates_final,
    "\n"
)


# PAM50 validity
pam50_valid <- all(
    step4$PAM50 %in%
        valid_labels
)


cat(
    "All PAM50 labels valid:",
    pam50_valid,
    "\n\n"
)


# ------------------------------------------------------------
# 43. HARD ASSERTIONS
# ------------------------------------------------------------

stopifnot(
    final_na == 0
)


stopifnot(
    !duplicate_symbols
)


stopifnot(
    sample_order_ok
)


stopifnot(
    !patient_duplicates_final
)


stopifnot(
    pam50_valid
)


# ------------------------------------------------------------
# 44. SAVE FINAL EXPRESSION MATRIX
# ------------------------------------------------------------

final_expression_file <- file.path(
    EXPR_DIR,
    "TCGA_expression_clean.csv"
)


write.csv(
    log_tpm_df,
    final_expression_file
)


cat(
    "Final expression matrix saved to:\n",
    final_expression_file,
    "\n\n"
)


# ------------------------------------------------------------
# 45. SAVE FINAL PAM50 FILE
# ------------------------------------------------------------

final_pam50 <- step4 %>%

    select(
        patient_id,
        full_barcode,
        PAM50
    )


final_pam50_file <- file.path(
    PAM50_DIR,
    "TCGA_PAM50.csv"
)


write.csv(
    final_pam50,
    final_pam50_file,
    row.names = FALSE
)


# ------------------------------------------------------------
# 46. SAVE FINAL METADATA
# ------------------------------------------------------------

final_metadata_file <- file.path(
    META_DIR,
    "TCGA_metadata.csv"
)


write.csv(
    step4,
    final_metadata_file,
    row.names = FALSE
)


# ------------------------------------------------------------
# 47. SAVE FINAL RDS
# ------------------------------------------------------------

saveRDS(
    log_tpm_df,
    file.path(
        CLEAN_EXPR_DIR,
        "TCGA_expression_clean.rds"
    )
)


# ------------------------------------------------------------
# 48. FINAL DATASET SUMMARY
# ------------------------------------------------------------

final_summary <- data.frame(

    metric = c(
        "Final patients",
        "Final samples",
        "Final genes",
        "Expression transformation",
        "Low-expression TPM threshold",
        "Minimum sample fraction",
        "PAM50 classes"
    ),

    value = c(
        length(
            unique(
                step4$patient_id
            )
        ),

        ncol(log_tpm_df),

        nrow(log_tpm_df),

        "log2(TPM + 1)",

        expression_threshold,

        sample_fraction,

        paste(
            valid_labels,
            collapse = ", "
        )
    )

)


write.csv(
    final_summary,
    file.path(
        QC_DIR,
        "final_dataset_summary.csv"
    ),
    row.names = FALSE
)


# ------------------------------------------------------------
# 49. DATA PROVENANCE NOTE
# ------------------------------------------------------------

provenance_file <- file.path(
    QC_DIR,
    "DATA_PROVENANCE.txt"
)


sink(
    provenance_file
)


cat("TCGA-BRCA ML PROJECT\n")
cat("=====================\n\n")

cat(
    "Data source: GDC / TCGAbiolinks\n"
)

cat(
    "Project: TCGA-BRCA\n"
)

cat(
    "Expression workflow: STAR - Counts\n"
)

cat(
    "Sample selection: Primary Tumor only\n"
)

cat(
    "Patient-level deduplication: first retained primary tumour sample\n"
)

cat(
    "PAM50 classes: LumA, LumB, Her2, Basal, Normal\n"
)

cat(
    "Expression transformation: log2(TPM + 1)\n"
)

cat(
    "Low-expression filter: TPM > 1 in >= 20% of final samples\n"
)

cat(
    "Gene ID conversion: Ensembl -> gene symbol\n"
)

cat(
    "Duplicate gene symbols: highest mean log-expression retained\n"
)

cat(
    "Date/time processed:",
    as.character(Sys.time()),
    "\n\n"
)


cat("FINAL DATASET\n")
cat("-------------\n")

cat(
    "Patients:",
    length(
        unique(
            step4$patient_id
        )
    ),
    "\n"
)

cat(
    "Samples:",
    ncol(log_tpm_df),
    "\n"
)

cat(
    "Genes:",
    nrow(log_tpm_df),
    "\n\n"
)


cat("PAM50 DISTRIBUTION\n")
cat("------------------\n")

print(
    table(
        step4$PAM50
    )
)


cat("\nSESSION INFORMATION\n")
cat("--------------------\n")

print(
    sessionInfo()
)


sink()


# ------------------------------------------------------------
# 50. FINAL COMPLETION MESSAGE
# ------------------------------------------------------------

cat("\n")
cat("============================================================\n")
cat("PREPROCESSING COMPLETE\n")
cat("============================================================\n\n")

cat("FINAL OUTPUTS\n\n")

cat("Expression:\n")
cat("  ", final_expression_file, "\n\n")

cat("PAM50:\n")
cat("  ", final_pam50_file, "\n\n")

cat("Metadata:\n")
cat("  ", final_metadata_file, "\n\n")

cat("QC directory:\n")
cat("  ", QC_DIR, "\n\n")

cat("Important QC files:\n")
cat("  filter_log.csv\n")
cat("  barcode_id_table.csv\n")
cat("  pam50_distribution.csv\n")
cat("  pam50_distribution.png\n")
cat("  library_sizes.csv\n")
cat("  library_size_distribution.png\n")
cat("  missing_data_summary.csv\n")
cat("  gene_expression_filter_summary.csv\n")
cat("  final_dataset_summary.csv\n")
cat("  DATA_PROVENANCE.txt\n\n")

cat("============================================================\n")
```

