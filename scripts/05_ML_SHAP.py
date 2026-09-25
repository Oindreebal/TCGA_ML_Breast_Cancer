#!/usr/bin/env python3

"""
WEEK 4 — COMPLETE MACHINE-LEARNING PIPELINE
============================================

Classical ML:
    7 feature sets × 6 classifiers = 42 combinations

Feature sets:
    1. Unweighted Hallmark
    2. Network-weighted Hallmark
    3. Network-weighted Hallmark V2 (normalized)
    4. Network-weighted Hallmark V3A (degree)
    5. Network-weighted Hallmark V3B (betweenness)
    6. Network-weighted Hallmark V3C (eigenvector)
    7. Network-weighted Hallmark V3D (pagerank)

    NOTE: PCA was removed from this study (raw-expression PCA feature
    set + its leakage-safe in-pipeline PCA step). See
    PCA_REMOVAL_NOTES.md for what depended on it and how each
    dependency was resolved.

Classifiers:
    1. Logistic Regression
    2. Random Forest
    3. XGBoost
    4. SVM-RBF
    5. Elastic Net
    6. PLS 

Validation:
    Nested 5-fold outer CV
    3-fold inner CV

Metrics:
    Macro-F1
    Balanced Accuracy
    Multiclass AUROC

Additional:
    Confusion matrices
    Classification reports
    Wilcoxon weighted vs unweighted comparison
    Frozen final classical model
"""

# ============================================================
# IMPORTS
# ============================================================

from pathlib import Path
import json
import requests
import pickle
import sys
import warnings
import importlib

import numpy as np
import pandas as pd

from scipy.stats import wilcoxon

from sklearn.base import clone, BaseEstimator, ClassifierMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder, LabelBinarizer
from sklearn.impute import SimpleImputer
from sklearn.cross_decomposition import PLSRegression
from sklearn.utils.validation import check_is_fitted, check_array
from sklearn.utils.multiclass import unique_labels

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from sklearn.model_selection import (
    StratifiedKFold,
    GridSearchCV,
    cross_validate,
    cross_val_predict,
    train_test_split,
)

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    f1_score,
    balanced_accuracy_score,
    roc_auc_score,
)

warnings.filterwarnings("ignore")


# ============================================================
# 0B. DEPENDENCY / PACKAGE CHECK
# ============================================================
#
# Verifies every package this script touches ANYWHERE (classical
# pipeline + optional GNN branch + plotting) is importable, before
# any real work starts. Required packages missing -> hard stop with
# an install hint. Optional packages missing -> printed warning,
# script continues (matching the existing try/except around
# matplotlib/seaborn and torch/torch_geometric further down).

print("\n" + "=" * 70)
print("DEPENDENCY CHECK")
print("=" * 70)

_REQUIRED_PACKAGES = {
    "numpy": "numpy",
    "pandas": "pandas",
    "scipy": "scipy",
    "sklearn": "scikit-learn",
    "xgboost": "xgboost",
    "joblib": "joblib",
    "requests": "requests",
}

_OPTIONAL_PACKAGES = {
    "matplotlib": "matplotlib",
    "seaborn": "seaborn",
    "torch": "torch",
    "torch_geometric": "torch-geometric",
    "networkx": "networkx",
    "statsmodels": "statsmodels",
    "shap": "shap",
}

_missing_required = []
_missing_optional = []

for _module_name, _pip_name in _REQUIRED_PACKAGES.items():
    try:
        _mod = importlib.import_module(_module_name)
        _ver = getattr(_mod, "__version__", "unknown")
        print(f"  [OK]       {_module_name:<16} (v{_ver})  required")
    except ImportError:
        _missing_required.append(_pip_name)
        print(f"  [MISSING]  {_module_name:<16} required")

for _module_name, _pip_name in _OPTIONAL_PACKAGES.items():
    try:
        _mod = importlib.import_module(_module_name)
        _ver = getattr(_mod, "__version__", "unknown")
        print(f"  [OK]       {_module_name:<16} (v{_ver})  optional")
    except ImportError:
        _missing_optional.append(_pip_name)
        print(f"  [MISSING]  {_module_name:<16} optional "
              f"(plots/GNN branch will be skipped)")

if _missing_required:
    raise ImportError(
        "\nMissing REQUIRED packages: "
        f"{', '.join(_missing_required)}\n"
        "Install with:\n"
        f"    pip install {' '.join(_missing_required)}\n"
        "or, if using conda:\n"
        f"    conda install -c conda-forge {' '.join(_missing_required)}\n"
    )

if _missing_optional:
    print(
        "\nWARNING: optional packages missing -> "
        f"{', '.join(_missing_optional)}"
    )
    print(
        "The classical ML comparison (Hallmark feature sets x "
        "Logistic/RandomForest/XGBoost/SVM/PLS-DA/ElasticNet) will still "
        "run and produce the concise comparison CSV. Only confusion-matrix "
    )
else:
    print("\nAll required and optional packages are present.")

print("Python executable:", sys.executable)
print("Python version:", sys.version.split()[0])


# ============================================================
# CONFIGURATION
# ============================================================

RANDOM_STATE = 42

N_OUTER = 5
N_INNER = 3

PROJECT = Path.home() / "TCGA_ML"

DATA_DIR = PROJECT / "01_data"
FEATURE_DIR = PROJECT / "05_features"

ML_DIR = PROJECT / "06_ML"

VALIDATION_DIR = (
    PROJECT /
    "07_validation" /
    "TCGA_nested_CV"
)

SHAP_DIR = PROJECT / "08_SHAP"

GNN_DIR = ML_DIR / "GNN"

for directory in [
    ML_DIR,
    VALIDATION_DIR,
    SHAP_DIR,
    GNN_DIR,
]:
    directory.mkdir(
        parents=True,
        exist_ok=True
    )


# ============================================================
# INPUT FILES
# ============================================================

UNWEIGHTED_FILE = (
    FEATURE_DIR /
    "Hallmark_unweighted.csv"
)

WEIGHTED_FILE = (
    FEATURE_DIR /
    "Hallmark_network_weighted.csv"
)

WEIGHTED_V2_FILE = (
    FEATURE_DIR /
    "Hallmark_network_weighted_v2.csv"
)

WEIGHTED_V3A_FILE = (
    FEATURE_DIR /
    "Hallmark_network_weighted_v3a_degree.csv"
)

WEIGHTED_V3B_FILE = (
    FEATURE_DIR /
    "Hallmark_network_weighted_v3b_betweenness.csv"
)
WEIGHTED_V3C_FILE = (
    FEATURE_DIR /
    "Hallmark_network_weighted_v3c_eigenvector.csv"
)

WEIGHTED_V3D_FILE = (
    FEATURE_DIR /
    "Hallmark_network_weighted_v3d_pagerank.csv"
)

LABEL_FILE = (
    DATA_DIR /
    "TCGA_PAM50" /
    "TCGA_PAM50.csv"
)

GPCM_FILE = (
    PROJECT /
    "04_STRING" /
    "pathway_networks" /
    "GPCM_final.gpickle"
)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def require_file(path):

    if not path.exists():

        raise FileNotFoundError(
            "\nRequired file not found:\n"
            f"    {path}\n\n"
            "Please check the project directory and filename."
        )


def normalize_tcga_id(value):
    """
    Convert TCGA sample IDs to patient-level IDs.

    Example:

        TCGA-XX-YYYY-01A
        ->
        TCGA-XX-YYYY
    """

    value = str(value).strip()

    if value.startswith("TCGA-"):

        parts = value.split("-")

        if len(parts) >= 3:

            return "-".join(parts[:3])

    return value


def normalize_id_series(series):

    return (
        series
        .astype(str)
        .map(normalize_tcga_id)
    )


def check_patient_ids(df, name):

    if "patient_id" not in df.columns:

        raise ValueError(
            f"{name} does not contain "
            "'patient_id'."
        )

    df["patient_id"] = (
        normalize_id_series(
            df["patient_id"]
        )
    )

    if df["patient_id"].duplicated().any():

        duplicated = (
            df.loc[
                df["patient_id"].duplicated(),
                "patient_id"
            ]
            .tolist()
        )

        raise ValueError(
            f"\n{name} contains duplicated "
            "patient IDs.\n"
            f"Examples: {duplicated[:10]}"
        )


def align_feature_dataframe(
    df,
    labels,
    name
):

    df = df.copy()

    check_patient_ids(
        df,
        name
    )

    label_ids = (
        normalize_id_series(
            labels["patient_id"]
        )
    )

    feature_ids = set(
        df["patient_id"]
    )

    label_id_set = set(
        label_ids
    )

    missing = sorted(
        label_id_set -
        feature_ids
    )

    extra = sorted(
        feature_ids -
        label_id_set
    )

    if missing:

        raise ValueError(
            f"\n{name} is missing "
            f"{len(missing)} patients.\n"
            f"Examples: {missing[:10]}"
        )

    if extra:

        print(
            f"WARNING: {name} contains "
            f"{len(extra)} extra patients."
        )

        print(
            "Those patients will be removed."
        )

    df = df[
        df["patient_id"].isin(
            label_ids
        )
    ].copy()

    df = (
        df
        .set_index("patient_id")
        .loc[label_ids]
        .reset_index()
    )

    if not np.array_equal(
        df["patient_id"].values,
        label_ids.values
    ):

        raise AssertionError(
            f"{name} failed patient alignment."
        )

    return df


# NOTE: load_expression_patient_by_gene() was removed along with PCA.
# It only ever loaded raw gene expression for the PCA feature set
# (which used raw expression + an in-pipeline PCA() step). With PCA
# removed from the study, nothing else in this file consumes raw
# per-gene expression, so the loader and its EXPR_FILE input were
# removed together rather than left as dead code.


# ============================================================
# 1. CHECK INPUT FILES
# ============================================================

print("\n" + "=" * 70)
print("WEEK 4 — INPUT VALIDATION")
print("=" * 70)

required_files = [
    UNWEIGHTED_FILE,
    WEIGHTED_FILE,
    WEIGHTED_V2_FILE,
    WEIGHTED_V3A_FILE,
    WEIGHTED_V3B_FILE,
    WEIGHTED_V3C_FILE,
    WEIGHTED_V3D_FILE,
    LABEL_FILE,
    GPCM_FILE,
]

for file in required_files:

    require_file(file)


# ============================================================
# 2. LOAD DATA
# ============================================================

unweighted = pd.read_csv(
    UNWEIGHTED_FILE
)

weighted = pd.read_csv(
    WEIGHTED_FILE
)

weighted_v2 = pd.read_csv(
    WEIGHTED_V2_FILE
)

weighted_v3a = pd.read_csv(
    WEIGHTED_V3A_FILE
)

weighted_v3b = pd.read_csv(
    WEIGHTED_V3B_FILE
)

weighted_v3c = pd.read_csv(
    WEIGHTED_V3C_FILE
)

weighted_v3d = pd.read_csv(
    WEIGHTED_V3D_FILE
)

labels = pd.read_csv(
    LABEL_FILE
)

print(
    "Unweighted Hallmark:",
    unweighted.shape
)

print(
    "Network-weighted Hallmark:",
    weighted.shape
)

print(
    "Network-weighted Hallmark V2:",
    weighted_v2.shape
)

print(
    "Network-weighted Hallmark V3A:",
    weighted_v3a.shape
)

print(
    "Network-weighted Hallmark V3B:",
    weighted_v3b.shape
)

print(
    "Network-weighted Hallmark V3C:",
    weighted_v3c.shape
)

print(
    "Network-weighted Hallmark V3D:",
    weighted_v3d.shape
)

print(
    "Labels:",
    labels.shape
)


# ============================================================
# 3. VALIDATE LABEL FILE
# ============================================================

if "patient_id" not in labels.columns:

    raise ValueError(
        "TCGA_PAM50.csv must contain "
        "'patient_id'."
    )

if "PAM50" not in labels.columns:

    raise ValueError(
        "TCGA_PAM50.csv must contain "
        "'PAM50'."
    )

labels["patient_id"] = (
    normalize_id_series(
        labels["patient_id"]
    )
)

check_patient_ids(
    labels,
    "TCGA_PAM50"
)


# ============================================================
# 4. ALIGN FEATURE MATRICES
# ============================================================

unweighted = align_feature_dataframe(
    unweighted,
    labels,
    "Hallmark_unweighted"
)

weighted = align_feature_dataframe(
    weighted,
    labels,
    "Hallmark_network_weighted"
)

weighted_v2 = align_feature_dataframe(
    weighted_v2,
    labels,
    "Hallmark_network_weighted_v2"
)

weighted_v3a = align_feature_dataframe(
    weighted_v3a,
    labels,
    "Hallmark_network_weighted_v3a"
)

weighted_v3b = align_feature_dataframe(
    weighted_v3b,
    labels,
    "Hallmark_network_weighted_v3b"
)

weighted_v3c = align_feature_dataframe(
    weighted_v3c,
    labels,
    "Hallmark_network_weighted_v3c"
)

weighted_v3d = align_feature_dataframe(
    weighted_v3d,
    labels,
    "Hallmark_network_weighted_v3d"
)


# ============================================================
# 5. PAM50 DISTRIBUTION
# ============================================================

print("\nPAM50 distribution:")

print(
    labels["PAM50"]
    .value_counts()
)


# ============================================================
# 6. FEATURE COUNT CHECK
# ============================================================

unweighted_features = [
    c
    for c in unweighted.columns
    if c != "patient_id"
]

weighted_features = [
    c
    for c in weighted.columns
    if c != "patient_id"
]

weighted_v2_features = [
    c
    for c in weighted_v2.columns
    if c != "patient_id"
]

weighted_v3a_features = [
    c
    for c in weighted_v3a.columns
    if c != "patient_id"
]

weighted_v3b_features = [
    c
    for c in weighted_v3b.columns
    if c != "patient_id"
]

weighted_v3c_features = [
    c
    for c in weighted_v3c.columns
    if c != "patient_id"
]

weighted_v3d_features = [
    c
    for c in weighted_v3d.columns
    if c != "patient_id"
]

print("\nFeature counts:")

print(
    "Unweighted Hallmark:",
    len(unweighted_features)
)

print(
    "Network-weighted Hallmark:",
    len(weighted_features)
)

print(
    "Network-weighted Hallmark V2:",
    len(weighted_v2_features)
)

print(
    "Network-weighted Hallmark V3A:",
    len(weighted_v3a_features)
)

print(
    "Network-weighted Hallmark V3B:",
    len(weighted_v3b_features)
)

print(
    "Network-weighted Hallmark V3C:",
    len(weighted_v3c_features)
)

print(
    "Network-weighted Hallmark V3D:",
    len(weighted_v3d_features)
)

if len(unweighted_features) != 50:

    raise ValueError(
        "Hallmark_unweighted.csv should contain "
        "exactly 50 pathway features."
    )

if len(weighted_features) != 50:

    raise ValueError(
        "Hallmark_network_weighted.csv should contain "
        "exactly 50 pathway features."
    )

for _wname, _wfeats in [
    ("Hallmark_network_weighted_v2.csv", weighted_v2_features),
    ("Hallmark_network_weighted_v3a.csv", weighted_v3a_features),
    ("Hallmark_network_weighted_v3b.csv", weighted_v3b_features),
    ("Hallmark_network_weighted_v3c.csv", weighted_v3c_features),
    ("Hallmark_network_weighted_v3d.csv", weighted_v3d_features),
]:

    if len(_wfeats) != 50:

        raise ValueError(
            f"{_wname} should contain "
            "exactly 50 pathway features."
        )


# ============================================================
# 7. ENCODE PAM50 LABELS
# ============================================================
#
# NOTE: section was previously "7. RAW EXPRESSION" (loading raw
# per-patient gene expression for the PCA feature set) followed by
# "8. ENCODE PAM50 LABELS". With PCA removed, raw expression loading
# was removed too (see PCA_REMOVAL_NOTES.md) and this section
# renumbered accordingly.

print("\n" + "=" * 70)
print("ENCODING PAM50 LABELS")
print("=" * 70)

label_encoder = LabelEncoder()

y = label_encoder.fit_transform(
    labels["PAM50"].astype(str)
)

class_names = (
    label_encoder.classes_
    .tolist()
)

print(
    "Classes:",
    class_names
)

print(
    "Encoded labels:",
    list(
        range(
            len(class_names)
        )
    )
)


# ============================================================
# 8. FEATURE SETS
# ============================================================

feature_sets = {

    "Unweighted_Hallmark":
        unweighted.drop(
            columns=[
                "patient_id"
            ]
        ),

    "Network_Weighted_Hallmark":
        weighted.drop(
            columns=[
                "patient_id"
            ]
        ),

    "Network_Weighted_Hallmark_V2":
        weighted_v2.drop(
            columns=[
                "patient_id"
            ]
        ),

    "Network_Weighted_Hallmark_V3A":
        weighted_v3a.drop(
            columns=[
                "patient_id"
            ]
        ),

    "Network_Weighted_Hallmark_V3B":
        weighted_v3b.drop(
            columns=[
                "patient_id"
            ]
        ),

    "Network_Weighted_Hallmark_V3C":
        weighted_v3c.drop(
            columns=[
                "patient_id"
            ]
        ),

    "Network_Weighted_Hallmark_V3D":
        weighted_v3d.drop(
            columns=[
                "patient_id"
            ]
        ),
}


for name in feature_sets:

    feature_sets[name] = (
        feature_sets[name]
        .apply(
            pd.to_numeric,
            errors="coerce"
        )
    )


# ============================================================
# 10. XGBOOST
# ============================================================

try:

    from xgboost import XGBClassifier

except ImportError as error:

    raise ImportError(
        "\nXGBoost is not installed.\n\n"
        "Run:\n"
        "conda activate bioinfo\n"
        "conda install -c conda-forge xgboost\n"
    ) from error


# ============================================================
# 10B. PLS-DA CLASSIFIER (custom sklearn-compatible wrapper)
# ============================================================
#
# scikit-learn has no built-in PLS-DA classifier -- PLSRegression is a
# regressor. This wraps it into a proper classifier for the multiclass
# PAM50 problem, following the standard PLS-DA recipe:
#
#   1. fit(X, y): one-hot encode y into a dummy-variable matrix Y
#      (n_samples x n_classes), fit PLSRegression(n_components) to
#      predict Y from X.
#   2. predict(X): PLSRegression.predict(X) gives a continuous score
#      per class; take argmax.
#   3. predict_proba(X): softmax the continuous per-class scores into
#      something that behaves like a probability vector (sums to 1,
#      higher score = higher probability). This is what lets
#      roc_auc_ovr scoring work in cross_validate/GridSearchCV exactly
#      like it does for the other 4 classifiers.
#
# Fully fold-safe: fit() is only ever called by the sklearn Pipeline
# inside GridSearchCV/cross_validate on the current training fold, no
# different from how LogisticRegression/RandomForest/etc. are handled.

class PLSDAClassifier(ClassifierMixin, BaseEstimator):
    """
    PLS Discriminant Analysis for multiclass classification.

    NOTE ON CLASS ORDER: ClassifierMixin MUST come before BaseEstimator
    in this inheritance list. scikit-learn's estimator-tag system (the
    __sklearn_tags__ chain, sklearn >=1.6) resolves via Python's normal
    MRO, and ClassifierMixin.__sklearn_tags__ is what sets
    estimator_type="classifier". Writing this the other way around
    (BaseEstimator, ClassifierMixin) silently produces an estimator
    that sklearn's `is_classifier()` reports as NOT a classifier --
    which does not fail at fit() or predict() time, only later, when
    GridSearchCV/cross_validate tries to use roc_auc_ovr scoring and
    raises "Got a regressor with response_method=predict_proba". This
    was caught by testing the class through the exact same
    GridSearchCV + cross_validate + roc_auc_ovr path used by this
    pipeline before shipping it -- verify this stays fixed if this
    class is ever edited.
    """

    def __init__(self, n_components=5):
        self.n_components = n_components

    def fit(self, X, y):
        X = check_array(X)
        self.classes_ = unique_labels(y)

        # Guard against asking for more components than the data
        # supports (can happen inside small GridSearchCV inner folds).
        max_allowed = max(1, min(X.shape[0] - 1, X.shape[1]))
        n_components = min(self.n_components, max_allowed)

        self._label_binarizer_ = LabelBinarizer()
        Y = self._label_binarizer_.fit_transform(y)

        # LabelBinarizer collapses a 2-class problem to a single
        # column; PLSRegression needs at least 2 target columns to
        # behave as intended for a multiclass one-vs-all style score.
        if Y.shape[1] == 1:
            Y = np.hstack([1 - Y, Y])

        self.pls_ = PLSRegression(n_components=n_components)
        self.pls_.fit(X, Y)
        self.n_features_in_ = X.shape[1]
        return self

    def _decision_scores(self, X):
        check_is_fitted(self, "pls_")
        X = check_array(X)
        return self.pls_.predict(X)

    def predict(self, X):
        scores = self._decision_scores(X)
        indices = np.argmax(scores, axis=1)
        return self.classes_[indices]

    def predict_proba(self, X):
        scores = self._decision_scores(X)
        # Numerically stable softmax across the class axis.
        shifted = scores - scores.max(axis=1, keepdims=True)
        exp_scores = np.exp(shifted)
        return exp_scores / exp_scores.sum(axis=1, keepdims=True)


# ============================================================
# 11. MODELS
# ============================================================

models_and_grids = {

    "Logistic": (

        LogisticRegression(
            penalty="l1",
            solver="saga",
            max_iter=5000,
            random_state=RANDOM_STATE,
        ),

        {
            "clf__C": [
                0.01,
                0.1,
                1,
                10,
            ]
        },
    ),

    "RandomForest": (

        RandomForestClassifier(
            n_estimators=300,
            random_state=RANDOM_STATE,
            n_jobs=1,
            class_weight="balanced",
        ),

        {
            "clf__n_estimators": [
                200,
                500,
            ],

            "clf__max_depth": [
                None,
                5,
                10,
            ],

            "clf__min_samples_leaf": [
                1,
                3,
                5,
            ],
        },
    ),

    "XGBoost": (

        XGBClassifier(
            objective="multi:softprob",
            eval_metric="mlogloss",
            num_class=len(
                class_names
            ),
            random_state=RANDOM_STATE,
            n_jobs=1,
            tree_method="hist",
        ),

        {
            "clf__n_estimators": [
                200,
                500,
            ],

            "clf__max_depth": [
                3,
                5,
                7,
            ],

            "clf__learning_rate": [
                0.01,
                0.1,
            ],
        },
    ),

    "SVM_RBF": (

        SVC(
            kernel="rbf",
            probability=True,
            random_state=RANDOM_STATE,
        ),

        {
            "clf__C": [
                0.1,
                1,
                10,
            ],

            "clf__gamma": [
                "scale",
                0.01,
                0.1,
            ],
        },
    ),

    "PLS_DA": (

        PLSDAClassifier(),

        {
            # Capped at 10: safe across every feature set in play,
            # including the 50-column Hallmark pathway matrices
            # (n_components must be <= n_features and <= n_samples-1
            # inside every inner-CV training fold; the classifier's
            # own fit() also clips defensively as a second guard).
            "clf__n_components": [
                2,
                5,
                10,
            ],
        },
    ),

    "ElasticNet": (

        LogisticRegression(
            penalty="elasticnet",
            solver="saga",
            max_iter=5000,
            random_state=RANDOM_STATE,
        ),

        {
            "clf__C": [
                0.01,
                0.1,
                1,
                10,
            ],

            "clf__l1_ratio": [
                0.1,
                0.5,
                0.9,
            ],
        },
    ),
}


# ============================================================
# 12. LEAKAGE-SAFE PIPELINE
# ============================================================

def build_pipeline(
    feature_set_name,
    model,
    n_training_samples=None,
    n_features=None
):
    # NOTE: n_training_samples/n_features are no longer used inside this
    # function -- they only ever fed the PCA branch's dynamic
    # n_components calculation. Signature kept unchanged so existing
    # call sites elsewhere in this file don't need to be touched; see
    # PCA_REMOVAL_NOTES.md.

    steps = [

        (
            "imputer",
            SimpleImputer(
                strategy="median"
            ),
        ),

        (
            "scaler",
            StandardScaler()
        ),
    ]

    steps.append(
        (
            "clf",
            clone(model)
        )
    )

    return Pipeline(
        steps
    )


# ============================================================
# 13. CROSS-VALIDATION OBJECTS
# ============================================================

outer_cv = StratifiedKFold(
    n_splits=N_OUTER,
    shuffle=True,
    random_state=RANDOM_STATE,
)

inner_cv = StratifiedKFold(
    n_splits=N_INNER,
    shuffle=True,
    random_state=RANDOM_STATE,
)


scoring = {

    "macro_f1":
        "f1_macro",

    "balanced_accuracy":
        "balanced_accuracy",

    "roc_auc_ovr":
        "roc_auc_ovr",
}

# ============================================================
# 14. NESTED CV
# ============================================================

print("\n" + "=" * 70)
print("STARTING NESTED 5 × 3 CROSS-VALIDATION")
print("=" * 70)

# ------------------------------------------------------------
# Output file for checkpointing
# ------------------------------------------------------------

raw_results_file = (
    VALIDATION_DIR /
    "nested_cv_results_raw.csv"
)

# Make sure the validation directory exists
VALIDATION_DIR.mkdir(
    parents=True,
    exist_ok=True
)

# ------------------------------------------------------------
# Resume from previous completed combinations if available
# ------------------------------------------------------------

results = []

if raw_results_file.exists():

    print(
        f"\nExisting checkpoint found:\n"
        f"{raw_results_file}"
    )

    previous_results_df = pd.read_csv(
        raw_results_file
    )

    # Only consider a feature-set/model combination
    # complete when all outer folds are present.
    fold_counts = (
        previous_results_df
        .groupby(
            ["feature_set", "model"]
        )
        .size()
    )

    completed_combinations = {
        combination
        for combination, count in fold_counts.items()
        if count == N_OUTER
    }

    results = (
        previous_results_df
        .to_dict(orient="records")
    )

    print(
        f"Loaded {len(completed_combinations)} "
        f"completed combinations from checkpoint."
    )

else:

    completed_combinations = set()

    print(
        "\nNo previous checkpoint found."
    )

# ------------------------------------------------------------
# Nested CV loop
# ------------------------------------------------------------

for feature_set_name, X in feature_sets.items():

    print("\n" + "-" * 70)
    print(
        f"FEATURE SET: "
        f"{feature_set_name}"
    )
    print("-" * 70)

    for model_name, (
        model,
        parameter_grid
    ) in models_and_grids.items():

        combination = (
            feature_set_name,
            model_name
        )

        # ----------------------------------------------------
        # Skip combinations that were already completed
        # ----------------------------------------------------

        if combination in completed_combinations:

            print(
                f"\nSKIPPING already completed: "
                f"{feature_set_name} + "
                f"{model_name}"
            )

            continue

        print(
            f"\nRunning: "
            f"{feature_set_name} + "
            f"{model_name}"
        )

        # ----------------------------------------------------
        # Shape information
        #
        # Kept for build_pipeline() compatibility.
        # No PCA is used anywhere here.
        # ----------------------------------------------------

        n_samples = X.shape[0]
        n_features = X.shape[1]

        pipeline = build_pipeline(
            feature_set_name,
            model,
            n_training_samples=n_samples,
            n_features=n_features
        )

        # ----------------------------------------------------
        # Inner CV: hyperparameter tuning
        # ----------------------------------------------------

        search = GridSearchCV(

            estimator=pipeline,

            param_grid=parameter_grid,

            cv=inner_cv,

            scoring="f1_macro",

            n_jobs=-1,

            refit=True,

            error_score="raise",
        )

        # ----------------------------------------------------
        # Outer CV: unbiased performance estimation
        # ----------------------------------------------------

        try:

            cv_results = cross_validate(

                search,

                X,

                y,

                cv=outer_cv,

                scoring=scoring,

                return_estimator=True,

                n_jobs=1,

                error_score="raise",
            )

        except Exception as error:

            # ------------------------------------------------
            # Save everything completed BEFORE the failure
            # ------------------------------------------------

            checkpoint_df = pd.DataFrame(
                results
            )

            checkpoint_df.to_csv(
                raw_results_file,
                index=False
            )

            print(
                "\n" + "!" * 70
            )

            print(
                "ERROR during nested CV"
            )

            print(
                f"Feature set: {feature_set_name}"
            )

            print(
                f"Model: {model_name}"
            )

            print(
                f"Completed results saved to:\n"
                f"{raw_results_file}"
            )

            print(
                "!" * 70
            )

            # Re-raise the original error so the
            # traceback remains visible.
            raise error

        # ----------------------------------------------------
        # Store outer-fold results
        # ----------------------------------------------------

        for fold_idx in range(N_OUTER):

            estimator = (
                cv_results[
                    "estimator"
                ][fold_idx]
            )

            best_params = (
                estimator
                .best_params_
            )

            results.append(
                {

                    "feature_set":
                        feature_set_name,

                    "model":
                        model_name,

                    "fold":
                        fold_idx + 1,

                    "macro_f1":
                        cv_results[
                            "test_macro_f1"
                        ][fold_idx],

                    "balanced_accuracy":
                        cv_results[
                            "test_balanced_accuracy"
                        ][fold_idx],

                    "roc_auc_ovr":
                        cv_results[
                            "test_roc_auc_ovr"
                        ][fold_idx],

                    "best_params":
                        json.dumps(
                            best_params,
                            sort_keys=True,
                            default=str
                        ),
                }
            )

        # ----------------------------------------------------
        # Mark combination as completed
        # ----------------------------------------------------

        completed_combinations.add(
            combination
        )

        # ----------------------------------------------------
        # CHECKPOINT AFTER EVERY COMPLETED COMBINATION
        # ----------------------------------------------------

        results_df = pd.DataFrame(
            results
        )

        results_df.to_csv(
            raw_results_file,
            index=False
        )

        # ----------------------------------------------------
        # Print performance
        # ----------------------------------------------------

        print(
            f"\nCompleted: "
            f"{feature_set_name} + "
            f"{model_name}"
        )

        print(
            f"Macro-F1: "
            f"{cv_results['test_macro_f1'].mean():.4f} "
            f"+/- "
            f"{cv_results['test_macro_f1'].std():.4f}"
        )

        print(
            f"Balanced accuracy: "
            f"{cv_results['test_balanced_accuracy'].mean():.4f}"
        )

        print(
            f"AUROC: "
            f"{cv_results['test_roc_auc_ovr'].mean():.4f}"
        )

        print(
            f"Checkpoint saved to:\n"
            f"{raw_results_file}"
        )


# ============================================================
# 15. SAVE FINAL RAW CV RESULTS
# ============================================================

results_df = pd.DataFrame(
    results
)

results_df.to_csv(
    raw_results_file,
    index=False
)

print("\n" + "=" * 70)
print("NESTED CV COMPLETE")
print("=" * 70)

print(
    f"\nTotal completed combinations: "
    f"{len(completed_combinations)}"
)

print(
    f"Total outer-fold results: "
    f"{len(results_df)}"
)

print(
    "\nSaved:"
)

print(
    raw_results_file
)

# ============================================================
# 16. MODEL SUMMARY
# ============================================================

summary = (
    results_df
    .groupby(
        [
            "feature_set",
            "model"
        ]
    )
    .agg(

        mean_macro_f1=(
            "macro_f1",
            "mean"
        ),

        std_macro_f1=(
            "macro_f1",
            "std"
        ),

        mean_balanced_accuracy=(
            "balanced_accuracy",
            "mean"
        ),

        std_balanced_accuracy=(
            "balanced_accuracy",
            "std"
        ),

        mean_roc_auc_ovr=(
            "roc_auc_ovr",
            "mean"
        ),

        std_roc_auc_ovr=(
            "roc_auc_ovr",
            "std"
        ),
    )
    .reset_index()
    .sort_values(
        [
            "mean_macro_f1",
            "mean_balanced_accuracy",
        ],
        ascending=False
    )
)


summary_file = (
    VALIDATION_DIR /
    "model_comparison_summary.csv"
)

summary.to_csv(
    summary_file,
    index=False
)


print("\n" + "=" * 70)
print("MODEL COMPARISON")
print("=" * 70)

print(
    summary.to_string(
        index=False
    )
)


# ============================================================
# 16B. CONCISE ALL-COMBINATIONS SCORE TABLE
# ============================================================
#
# One row per (feature_set, model) combination -- every combination
# that ran (7 feature sets x 6 models = 42 rows after PCA removal --
# see PCA_REMOVAL_NOTES.md), each metric as "mean ± std" in
# a single readable cell, plus the underlying numeric columns kept
# alongside for anything downstream that wants to sort/filter/plot
# without re-parsing the formatted string.

def _fmt_mean_std(mean_val, std_val):
    return f"{mean_val:.4f} \u00b1 {std_val:.4f}"


concise = summary.copy()

concise["macro_f1"] = concise.apply(
    lambda r: _fmt_mean_std(r["mean_macro_f1"], r["std_macro_f1"]),
    axis=1,
)

concise["balanced_accuracy"] = concise.apply(
    lambda r: _fmt_mean_std(
        r["mean_balanced_accuracy"], r["std_balanced_accuracy"]
    ),
    axis=1,
)

concise["roc_auc_ovr"] = concise.apply(
    lambda r: _fmt_mean_std(r["mean_roc_auc_ovr"], r["std_roc_auc_ovr"]),
    axis=1,
)

concise = concise[
    [
        "feature_set",
        "model",
        "macro_f1",
        "balanced_accuracy",
        "roc_auc_ovr",
        "mean_macro_f1",
        "std_macro_f1",
        "mean_balanced_accuracy",
        "std_balanced_accuracy",
        "mean_roc_auc_ovr",
        "std_roc_auc_ovr",
    ]
].sort_values(
    "mean_macro_f1",
    ascending=False,
)

concise_file = (
    VALIDATION_DIR /
    "all_combinations_concise_scores.csv"
)

concise.to_csv(
    concise_file,
    index=False,
)

print(
    "\nSaved concise all-combinations score table:",
    concise_file
)

print(
    f"Total combinations: {len(concise)} "
    f"({len(feature_sets)} feature sets x "
    f"{len(models_and_grids)} models)"
)


# ============================================================
# 17. WILCOXON TEST
# ============================================================

print("\n" + "=" * 70)
print(
    "NETWORK-WEIGHTED VS "
    "UNWEIGHTED COMPARISON"
)
print("=" * 70)

wilcoxon_results = []

for model_name in models_and_grids:

    weighted_scores = (
        results_df[
            (
                results_df[
                    "feature_set"
                ]
                ==
                "Network_Weighted_Hallmark"
            )
            &
            (
                results_df[
                    "model"
                ]
                ==
                model_name
            )
        ]
        .sort_values("fold")
        ["macro_f1"]
        .values
    )

    unweighted_scores = (
        results_df[
            (
                results_df[
                    "feature_set"
                ]
                ==
                "Unweighted_Hallmark"
            )
            &
            (
                results_df[
                    "model"
                ]
                ==
                model_name
            )
        ]
        .sort_values("fold")
        ["macro_f1"]
        .values
    )

    try:

        statistic, p_value = (
            wilcoxon(
                weighted_scores,
                unweighted_scores,
                zero_method="wilcox",
                alternative="two-sided",
            )
        )

    except ValueError:

        statistic = np.nan
        p_value = np.nan

    wilcoxon_results.append(
        {

            "model":
                model_name,

            "weighted_mean_macro_f1":
                weighted_scores.mean(),

            "unweighted_mean_macro_f1":
                unweighted_scores.mean(),

            "mean_difference":
                (
                    weighted_scores
                    -
                    unweighted_scores
                ).mean(),

            "wilcoxon_statistic":
                statistic,

            "p_value":
                p_value,
        }
    )


wilcoxon_df = pd.DataFrame(
    wilcoxon_results
)

wilcoxon_file = (
    VALIDATION_DIR /
    "weighted_vs_unweighted_wilcoxon.csv"
)

wilcoxon_df.to_csv(
    wilcoxon_file,
    index=False
)

print(
    wilcoxon_df.to_string(
        index=False
    )
)


# ============================================================
# 18. CONFUSION MATRICES
# ============================================================

print("\n" + "=" * 70)
print(
    "GENERATING CONFUSION MATRICES"
)
print("=" * 70)

all_predictions = []

for feature_set_name, X in feature_sets.items():

    for model_name, (
        model,
        parameter_grid
    ) in models_and_grids.items():

        print(
            f"Predictions: "
            f"{feature_set_name} + "
            f"{model_name}"
        )

        pipeline = build_pipeline(
            feature_set_name,
            model,
            n_training_samples=X.shape[0],
            n_features=X.shape[1]
        )

        search = GridSearchCV(

            estimator=pipeline,

            param_grid=parameter_grid,

            cv=inner_cv,

            scoring="f1_macro",

            n_jobs=-1,

            refit=True,

            error_score="raise",
        )

        y_pred = cross_val_predict(

            search,

            X,

            y,

            cv=outer_cv,

            method="predict",

            n_jobs=1,
        )

        cm = confusion_matrix(

            y,

            y_pred,

            labels=np.arange(
                len(class_names)
            )
        )

        safe_name = (
            f"{feature_set_name}_"
            f"{model_name}"
        )

        cm_df = pd.DataFrame(

            cm,

            index=[
                f"True_{x}"
                for x in class_names
            ],

            columns=[
                f"Pred_{x}"
                for x in class_names
            ]
        )

        cm_df.to_csv(

            VALIDATION_DIR /
            f"confusion_{safe_name}.csv"
        )


        report = classification_report(

            y,

            y_pred,

            labels=np.arange(
                len(class_names)
            ),

            target_names=class_names,

            output_dict=True,

            zero_division=0
        )

        report_df = pd.DataFrame(
            report
        ).T

        report_df.to_csv(

            VALIDATION_DIR /
            f"classification_report_{safe_name}.csv"
        )


        all_predictions.append(

            pd.DataFrame(
                {

                    "patient_id":
                        labels[
                            "patient_id"
                        ],

                    "true_label":
                        labels[
                            "PAM50"
                        ],

                    "predicted_label":
                        [
                            class_names[
                                int(i)
                            ]
                            for i
                            in y_pred
                        ],

                    "feature_set":
                        feature_set_name,

                    "model":
                        model_name,
                }
            )
        )


predictions_df = pd.concat(
    all_predictions,
    ignore_index=True
)

predictions_file = (
    VALIDATION_DIR /
    "nested_cv_predictions.csv"
)

predictions_df.to_csv(
    predictions_file,
    index=False
)


# ============================================================
# 19. CONFUSION MATRIX PLOTS
# ============================================================

try:

    import matplotlib.pyplot as plt
    import seaborn as sns

    for feature_set_name in feature_sets:

        for model_name in models_and_grids:

            safe_name = (
                f"{feature_set_name}_"
                f"{model_name}"
            )

            cm_file = (
                VALIDATION_DIR /
                f"confusion_{safe_name}.csv"
            )

            cm_df = pd.read_csv(
                cm_file,
                index_col=0
            )

            plt.figure(
                figsize=(6, 5)
            )

            sns.heatmap(
                cm_df,
                annot=True,
                fmt="d",
                xticklabels=class_names,
                yticklabels=class_names,
                cmap="Blues",
            )

            plt.title(
                f"{feature_set_name} + "
                f"{model_name}"
            )

            plt.xlabel(
                "Predicted"
            )

            plt.ylabel(
                "True"
            )

            plt.tight_layout()

            plt.savefig(

                VALIDATION_DIR /
                f"confusion_{safe_name}.png",

                dpi=300,

                bbox_inches="tight"
            )

            plt.close()

except ImportError:

    print(
        "\nmatplotlib/seaborn unavailable."
    )

    print(
        "CSV confusion matrices were still generated."
    )


# ============================================================
# 20. SELECT WINNING CLASSICAL MODEL
# ============================================================

best_row = summary.iloc[0]

final_feature_set = (
    best_row["feature_set"]
)

final_model_name = (
    best_row["model"]
)

final_model, final_grid = (
    models_and_grids[
        final_model_name
    ]
)


print("\n" + "=" * 70)
print(
    "WINNING CLASSICAL MODEL"
)
print("=" * 70)

print(
    "Feature set:",
    final_feature_set
)

print(
    "Model:",
    final_model_name
)

print(
    "Nested CV macro-F1:",
    f"{best_row['mean_macro_f1']:.4f}"
)

print(
    "Balanced accuracy:",
    f"{best_row['mean_balanced_accuracy']:.4f}"
)

print(
    "AUROC:",
    f"{best_row['mean_roc_auc_ovr']:.4f}"
)


# ============================================================
# 21. FINAL HYPERPARAMETER SELECTION
# ============================================================

print("\n" + "=" * 70)
print(
    "FINAL HYPERPARAMETER SELECTION"
)
print("=" * 70)

final_pipeline = build_pipeline(

    final_feature_set,

    final_model,

    n_training_samples=
        feature_sets[
            final_feature_set
        ].shape[0],

    n_features=
        feature_sets[
            final_feature_set
        ].shape[1]
)

final_search = GridSearchCV(

    estimator=final_pipeline,

    param_grid=final_grid,

    cv=inner_cv,

    scoring="f1_macro",

    n_jobs=-1,

    refit=True,

    error_score="raise"
)

final_search.fit(

    feature_sets[
        final_feature_set
    ],

    y
)

print(
    "\nFinal hyperparameters:"
)

print(
    final_search.best_params_
)


# ============================================================
# 22. SAVE FROZEN MODEL
# ============================================================

import joblib

frozen_model_file = (
    ML_DIR /
    "frozen_final_model.joblib"
)

joblib.dump(

    final_search.best_estimator_,

    frozen_model_file
)


metadata_file = (
    ML_DIR /
    "frozen_model_metadata.txt"
)

with open(
    metadata_file,
    "w"
) as f:

    f.write(
        "WEEK 4 FROZEN CLASSICAL MODEL\n"
    )

    f.write(
        "=" * 60 +
        "\n"
    )

    f.write(
        f"Feature set: "
        f"{final_feature_set}\n"
    )

    f.write(
        f"Model: "
        f"{final_model_name}\n"
    )

    f.write(
        f"Nested CV mean macro-F1: "
        f"{best_row['mean_macro_f1']:.6f}\n"
    )

    f.write(
        f"Nested CV SD macro-F1: "
        f"{best_row['std_macro_f1']:.6f}\n"
    )

    f.write(
        f"Balanced accuracy: "
        f"{best_row['mean_balanced_accuracy']:.6f}\n"
    )

    f.write(
        f"AUROC OVR: "
        f"{best_row['mean_roc_auc_ovr']:.6f}\n"
    )

    f.write(
        "Final parameters: "
        +
        json.dumps(
            final_search.best_params_,
            sort_keys=True,
            default=str
        )
        +
        "\n"
    )

    f.write(
        f"Random state: "
        f"{RANDOM_STATE}\n"
    )

    f.write(
        "NO FURTHER CLASSICAL MODEL "
        "TUNING AFTER THIS POINT.\n"
    )


print(
    "\nFrozen model saved:"
)

print(
    frozen_model_file
)

print(
    metadata_file
)


# ============================================================
# 22B. SHAP INTERPRETABILITY (frozen classical model)
# ============================================================
#
# Explains the SAME frozen model saved in Section 22 -- no retraining,
# no re-tuning, computed on the same feature set/model combination that
# won Section 20's comparison. Explainer choice is model-specific where
# a fast, exact explainer exists (TreeExplainer for RandomForest/XGBoost,
# LinearExplainer for Logistic/ElasticNet); everything else (SVM_RBF,
# PLS_DA) falls back to model-agnostic KernelExplainer on a background
# + evaluation subsample, since KernelExplainer on the full cohort can
# take hours. If a model-specific explainer raises (this varies across
# shap library versions, e.g. LinearExplainer's multiclass support has
# changed across releases), the code falls back to KernelExplainer
# automatically rather than crashing the whole pipeline.

print("\n" + "=" * 70)
print("SHAP INTERPRETABILITY")
print("=" * 70)

try:
    import shap
except ImportError as error:
    print("\nSHAP is not installed -- skipping interpretability section.")
    print("Install with: pip install shap")
    print("Import error:", error)
    shap = None

if shap is not None:

    fitted_pipeline = final_search.best_estimator_
    preprocessing_steps = fitted_pipeline[:-1]
    fitted_classifier = fitted_pipeline.named_steps["clf"]

    X_raw_for_shap = feature_sets[final_feature_set]
    X_transformed = preprocessing_steps.transform(X_raw_for_shap)

    # All remaining feature sets (Unweighted/Weighted/V2/V3a-d Hallmark
    # pathway scores) have real, meaningful column names -- the PCA
    # branch that synthesized "PC1, PC2, ..." names is no longer needed
    # since PCA was removed (see PCA_REMOVAL_NOTES.md).
    shap_feature_names = list(X_raw_for_shap.columns)

    if len(shap_feature_names) != X_transformed.shape[1]:
        # Defensive fallback -- keeps the script running even if a future
        # feature set doesn't cleanly match either branch above.
        shap_feature_names = [
            f"feature_{i}" for i in range(X_transformed.shape[1])
        ]

    print("Winning combination:", final_feature_set, "+", final_model_name)
    print("SHAP input shape:", X_transformed.shape)

    TREE_BASED_MODELS = {"RandomForest", "XGBoost"}
    LINEAR_MODELS = {"Logistic", "ElasticNet"}

    def _kernel_explainer_fallback(model, X_data, reason):
        """Model-agnostic fallback used when no fast explainer applies,
        or when a model-specific explainer fails for this shap version."""

        print(f"Using shap.KernelExplainer ({reason}).")

        background_n = min(100, X_data.shape[0])
        background = shap.sample(
            X_data, background_n, random_state=RANDOM_STATE
        )

        kernel_explainer = shap.KernelExplainer(
            model.predict_proba, background
        )

        sample_n = min(300, X_data.shape[0])
        rng = np.random.default_rng(RANDOM_STATE)
        sample_idx = rng.choice(
            X_data.shape[0], size=sample_n, replace=False
        )
        X_sample = X_data[sample_idx]

        print(
            f"KernelExplainer running on {sample_n} of "
            f"{X_data.shape[0]} patients (subsampled for runtime)."
        )

        sample_values = kernel_explainer.shap_values(X_sample)

        return kernel_explainer, sample_values, X_sample, background_n, sample_n

    background_size = None
    kernel_sample_size = None

    if final_model_name in TREE_BASED_MODELS:
        try:
            print("Using shap.TreeExplainer (exact, fast).")
            explainer = shap.TreeExplainer(fitted_classifier)
            raw_shap_values = explainer.shap_values(X_transformed)
            X_transformed_for_plots = X_transformed
        except Exception as tree_error:
            print(f"TreeExplainer failed ({tree_error}).")
            explainer, raw_shap_values, X_transformed_for_plots, \
                background_size, kernel_sample_size = _kernel_explainer_fallback(
                    fitted_classifier, X_transformed,
                    "TreeExplainer failed for this model/shap version"
                )

    elif final_model_name in LINEAR_MODELS:
        try:
            print("Using shap.LinearExplainer (exact for linear models).")
            explainer = shap.LinearExplainer(fitted_classifier, X_transformed)
            raw_shap_values = explainer.shap_values(X_transformed)
            X_transformed_for_plots = X_transformed
        except Exception as linear_error:
            print(f"LinearExplainer failed ({linear_error}).")
            explainer, raw_shap_values, X_transformed_for_plots, \
                background_size, kernel_sample_size = _kernel_explainer_fallback(
                    fitted_classifier, X_transformed,
                    "LinearExplainer failed for this model/shap version"
                )

    else:
        explainer, raw_shap_values, X_transformed_for_plots, \
            background_size, kernel_sample_size = _kernel_explainer_fallback(
                fitted_classifier, X_transformed,
                f"no model-specific explainer exists for {final_model_name}"
            )

    # Normalize shap_values into a list-of-arrays-per-class regardless of
    # which shap version/explainer produced it.
    if isinstance(raw_shap_values, list):
        shap_values_per_class = raw_shap_values
    elif isinstance(raw_shap_values, np.ndarray) and raw_shap_values.ndim == 3:
        shap_values_per_class = [
            raw_shap_values[:, :, i] for i in range(raw_shap_values.shape[2])
        ]
    else:
        shap_values_per_class = [raw_shap_values]

    n_shap_classes = len(shap_values_per_class)
    print(f"SHAP values computed for {n_shap_classes} class(es).")

    matplotlib_available = "matplotlib" not in _missing_optional

    if matplotlib_available:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        try:
            shap.summary_plot(
                shap_values_per_class,
                X_transformed_for_plots,
                feature_names=shap_feature_names,
                class_names=(
                    class_names if n_shap_classes == len(class_names) else None
                ),
                show=False,
            )
            plt.tight_layout()
            plt.savefig(SHAP_DIR / "global_shap_summary.png", dpi=150)
            plt.close()
            print("Saved:", SHAP_DIR / "global_shap_summary.png")
        except Exception as plot_error:
            print(f"Could not generate global SHAP summary plot: {plot_error}")
    else:
        print("matplotlib not available -- skipping SHAP plots (CSV outputs still saved).")

    top10_rows = []

    for class_idx in range(n_shap_classes):

        class_label = (
            class_names[class_idx]
            if class_idx < len(class_names)
            else f"class_{class_idx}"
        )

        class_shap_values = shap_values_per_class[class_idx]
        mean_abs_shap = np.abs(class_shap_values).mean(axis=0)
        top10_idx = np.argsort(mean_abs_shap)[::-1][:10]

        for rank, feat_idx in enumerate(top10_idx, start=1):
            top10_rows.append({
                "subtype": class_label,
                "rank": rank,
                "feature": shap_feature_names[feat_idx],
                "mean_abs_shap": float(mean_abs_shap[feat_idx]),
            })

        if matplotlib_available:
            try:
                shap.summary_plot(
                    class_shap_values,
                    X_transformed_for_plots,
                    feature_names=shap_feature_names,
                    show=False,
                    max_display=15,
                )
                plt.title(f"SHAP summary — {class_label}")
                plt.tight_layout()
                plt.savefig(
                    SHAP_DIR / f"shap_summary_{class_label}.png", dpi=150
                )
                plt.close()
            except Exception as plot_error:
                print(
                    f"Could not generate SHAP summary plot for "
                    f"{class_label}: {plot_error}"
                )

    top10_df = pd.DataFrame(top10_rows)
    top10_file = SHAP_DIR / "top10_features_per_subtype.csv"
    top10_df.to_csv(top10_file, index=False)
    print("Saved:", top10_file)

    print("\nTop feature per subtype:")
    for class_label in top10_df["subtype"].unique():
        top_feature_row = (
            top10_df[top10_df["subtype"] == class_label]
            .sort_values("rank")
            .iloc[0]
        )
        print(
            f"  {class_label}: {top_feature_row['feature']} "
            f"(mean |SHAP| = {top_feature_row['mean_abs_shap']:.4f})"
        )

    shap_metadata_file = SHAP_DIR / "shap_metadata.txt"
    with open(shap_metadata_file, "w") as f:
        f.write("SHAP INTERPRETABILITY — FROZEN CLASSICAL MODEL\n")
        f.write("=" * 60 + "\n")
        f.write(f"Feature set: {final_feature_set}\n")
        f.write(f"Model: {final_model_name}\n")
        f.write(f"Explainer type: {type(explainer).__name__}\n")
        f.write(f"Number of features explained: {X_transformed.shape[1]}\n")
        f.write(f"Number of classes: {n_shap_classes}\n")
        if background_size is not None:
            f.write(
                f"NOTE: KernelExplainer used with a background sample of "
                f"{background_size} and evaluated on a subsample of "
                f"{kernel_sample_size} patients (not the full cohort), "
                f"for runtime reasons.\n"
            )
        f.write(f"Random state: {RANDOM_STATE}\n")

    print("Saved:", shap_metadata_file)

else:
    print(
        "\nSHAP section skipped -- install `shap` to enable "
        "interpretability output."
    )



