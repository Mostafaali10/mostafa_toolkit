# ==========================================
# GLOBAL CONSTANTS & REPRODUCIBILITY
# ==========================================
RANDOM_STATE = 42
DEFAULT_FIGSIZE = (10, 6)
DEFAULT_N_ESTIMATORS = 100
DEFAULT_MAX_ITER = 1000

# ==========================================
# STATISTICAL ANALYSIS & EDA CONSTANTS
# ==========================================
DEFAULT_MAX_CARDINALITY = 50
FONTSIZE_OVERVIEW_TABLE = "13px"
FONTSIZE_STATS_TABLE = "12px"
COLOR_RED_BAR = "#d65f5f"
COLOR_BLUE_BAR = "#6baed6"

# ==========================================
# VISUALIZATION CONSTANTS
# ==========================================
DEFAULT_EDA_N_COLS = 3
DEFAULT_OUTLIERS_N_COLS = 4
DEFAULT_CATEGORICALS_N_COLS = 3

DEFAULT_DIST_BINS = 30
COLOR_DIST_BLUE = '#3498db'
COLOR_EDGE_BLACK = 'black'
DEFAULT_ALPHA = 0.8
DEFAULT_TITLE_FONTSIZE = 12

GRID_LINESTYLE = '--'
GRID_ALPHA = 0.7

COLOR_OUTLIERS_GREEN = '#2ecc71'
BOXPLOT_WIDTH = 0.5
BOXPLOT_LINEWIDTH = 1.5

PALETTE_MAGMA = 'magma'
LABEL_ROTATION_DEGREES = 45

TARGET_DIST_FIGSIZE = (8, 5)
PALETTE_VIRIDIS = 'viridis'
HEADER_FONTSIZE = 14
TEXT_FONTSIZE = 10
COLOR_REGRESSION_ORANGE = 'darkorange'
DEFAULT_TARGET_BINS = 50

CORRELATION_FIGSIZE = (20, 16)
CMAP_SPECTRAL = 'Spectral_r'
CORR_LINEWIDTH = 0.5
CORR_TITLE_FONTSIZE = 20
CORR_ANNOT_SIZE = 10
CMAP_GREENS = 'Greens'
CMAP_REDS_R = 'Reds_r'
CORR_HEAD_COUNT = 10

# ==========================================
# EVALUATION CONSTANTS
# ==========================================
CONF_MATRIX_FIGSIZE = (6, 5)
CMAP_BLUES = 'Blues'

ROC_FIGSIZE = (8, 6)
COLOR_ROC_ORANGE = 'darkorange'
COLOR_ROC_NAVY = 'navy'
CMAP_SET1 = 'Set1'

# ==========================================
# DEEP LEARNING & CALLBACK CONSTANTS
# ==========================================
COLOR_LR_DECAY = '#f39c12'
COLOR_TRAIN_CURVE = '#2980b9'
COLOR_VAL_CURVE = '#e74c3c'
DASHBOARD_TITLE_FONTSIZE = 20
SUBPLOT_TITLE_FONTSIZE = 14
DASHBOARD_HEIGHT_FACTOR = 4
DEFAULT_CALLBACK_ATOL = 1e-4

# ==========================================
# MODEL OPTIMIZATION BENCHMARK CONSTANTS
# ==========================================
BENCHMARK_KNN_LIMIT = 50000
BENCHMARK_SVM_LIMIT = 20000
DEFAULT_N_ITERATIONS = 10
DEFAULT_MAX_DIFF = 0.05

# ==========================================
# DEPLOYMENT CONSTANTS
# ==========================================
DEFAULT_DEPLOYMENT_DIR = "deployment_assets"
