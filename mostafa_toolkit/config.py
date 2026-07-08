import os
import random
import logging
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf

from .constants import RANDOM_STATE, DEFAULT_FIGSIZE

# ==========================================
# 1. WARNINGS & LOGGER CONFIGURATION
# ==========================================
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel(logging.ERROR)

# ==========================================
# 2. PANDAS DISPLAY OPTIONS
# ==========================================
pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.4f' % x)

# ==========================================
# 3. PLOTTING STYLE & DEFAULT RESOLUTIONS
# ==========================================
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams['figure.figsize'] = DEFAULT_FIGSIZE

# ==========================================
# 4. REPRODUCIBILITY LOCKS (SEEDS)
# ==========================================
random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['PYTHONHASHSEED'] = str(RANDOM_STATE)
