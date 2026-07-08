"""
Mostafa Toolkit
===============

A complete Machine Learning toolkit for Data Science workflows.

Author: Mostafa Ali
Version: 0.1.0
"""

__version__ = "0.1.0"
__author__ = "Mostafa Ali"

# ==========================================================
# Import common libraries
# ==========================================================

from .imports import *

# ==========================================================
# Import package modules
# ==========================================================

from . import constants
from . import callbacks
from . import utils
from . import eda
from . import visualization
from . import evaluation
from . import optimization
from . import deployment

# ==========================================================
# Import toolkit functions
# ==========================================================

# EDA
from .eda import statistical_analysis

# Visualization
from .visualization import (
    plot_distributions,
    plot_outliers,
    plot_categoricals,
    plot_target_distribution,
    plot_correlation,
    plot_dynamic_learning_curves,
)

# Evaluation
from .evaluation import (
    evaluate_classification,
    evaluate_regression,
    plot_roc_auc,
)

# Optimization
from .optimization import (
    benchmark_baseline_models,
    custom_random_search,
    custom_random_search_regression,
    get_combined_feature_importance,
)

# Deployment
from .deployment import export_for_deployment

# Utilities
from .utils import reduce_memory_usage

# Callbacks
from .callbacks import (
    RegressionMetricCheckpoint,
    DynamicClassificationCheckpoint,
)

# ==========================================================
# Public API
# ==========================================================

__all__ = [

    # -----------------------------
    # Common Libraries
    # -----------------------------

    "pd",
    "np",
    "plt",
    "sns",

    # -----------------------------
    # Machine Learning Models
    # -----------------------------

    "LinearRegression",
    "LogisticRegression",

    "DecisionTreeClassifier",
    "DecisionTreeRegressor",

    "RandomForestClassifier",
    "RandomForestRegressor",

    "GradientBoostingClassifier",
    "GradientBoostingRegressor",

    "KNeighborsClassifier",
    "KNeighborsRegressor",

    "SVC",
    "SVR",

    "XGBClassifier",
    "XGBRegressor",

    # -----------------------------
    # EDA
    # -----------------------------

    "statistical_analysis",

    # -----------------------------
    # Visualization
    # -----------------------------

    "plot_distributions",
    "plot_outliers",
    "plot_categoricals",
    "plot_target_distribution",
    "plot_correlation",
    "plot_dynamic_learning_curves",

    # -----------------------------
    # Evaluation
    # -----------------------------

    "evaluate_classification",
    "evaluate_regression",
    "plot_roc_auc",

    # -----------------------------
    # Optimization
    # -----------------------------

    "benchmark_baseline_models",
    "custom_random_search",
    "custom_random_search_regression",
    "get_combined_feature_importance",

    # -----------------------------
    # Deployment
    # -----------------------------

    "export_for_deployment",

    # -----------------------------
    # Utilities
    # -----------------------------

    "reduce_memory_usage",

    # -----------------------------
    # Callbacks
    # -----------------------------

    "RegressionMetricCheckpoint",
    "DynamicClassificationCheckpoint",
]