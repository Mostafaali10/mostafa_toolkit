# ==========================================
# 1. STANDARD LIBRARIES & WARNINGS
# ==========================================
import os
import logging
import gc
import random
import math
import warnings
import joblib
import shap
import sys
import copy
import time
import datetime
from pathlib import Path
from tqdm.auto import tqdm

# ==========================================
# 2. CORE DATA SCIENCE & VISUALIZATION
# ==========================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display, Markdown

# ==========================================
# 3. PREPROCESSING & PIPELINES
# ==========================================
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold, RandomizedSearchCV, ParameterSampler
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, OneHotEncoder, OrdinalEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from imblearn.pipeline import Pipeline 
from imblearn.over_sampling import SMOTE

# ==========================================
# 4. MACHINE LEARNING MODELS
# ==========================================
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from xgboost import XGBClassifier, XGBRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.naive_bayes import GaussianNB
import lightgbm as lgb
import catboost as cb

# ==========================================
# 5. DEEP LEARNING (TENSORFLOW/KERAS)
# ==========================================
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization, Concatenate, Conv2D, MaxPooling2D, Flatten, LSTM, GRU, Embedding, LeakyReLU, ReLU
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard, Callback
from tensorflow.keras.optimizers import Adam

# ==========================================
# 6. EVALUATION METRICS
# ==========================================
from sklearn.metrics import (classification_report, confusion_matrix, accuracy_score, 
                             precision_score, recall_score, f1_score, roc_curve, auc,
                             mean_absolute_error, mean_squared_error, r2_score)