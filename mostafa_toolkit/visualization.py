import gc
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display

from .constants import (
    DEFAULT_EDA_N_COLS,
    DEFAULT_OUTLIERS_N_COLS,
    DEFAULT_CATEGORICALS_N_COLS,
    DEFAULT_DIST_BINS,
    COLOR_DIST_BLUE,
    COLOR_EDGE_BLACK,
    DEFAULT_ALPHA,
    DEFAULT_TITLE_FONTSIZE,
    GRID_LINESTYLE,
    GRID_ALPHA,
    COLOR_OUTLIERS_GREEN,
    BOXPLOT_WIDTH,
    BOXPLOT_LINEWIDTH,
    PALETTE_MAGMA,
    LABEL_ROTATION_DEGREES,
    TARGET_DIST_FIGSIZE,
    PALETTE_VIRIDIS,
    HEADER_FONTSIZE,
    TEXT_FONTSIZE,
    COLOR_REGRESSION_ORANGE,
    DEFAULT_TARGET_BINS,
    CORRELATION_FIGSIZE,
    CMAP_SPECTRAL,
    CORR_LINEWIDTH,
    CORR_TITLE_FONTSIZE,
    CORR_ANNOT_SIZE,
    CMAP_GREENS,
    CMAP_REDS_R,
    CORR_HEAD_COUNT,
    COLOR_LR_DECAY,
    COLOR_TRAIN_CURVE,
    COLOR_VAL_CURVE,
    DASHBOARD_TITLE_FONTSIZE,
    SUBPLOT_TITLE_FONTSIZE,
    DASHBOARD_HEIGHT_FACTOR
)

def plot_distributions(df, columns=None, n_cols=DEFAULT_EDA_N_COLS): 
    """Plots distributions with modern styling and memory-safe binning."""
    if columns is None: 
        columns = df.select_dtypes(include=[np.number]).columns.tolist() 
 
    if len(columns) == 0: return 
 
    n_rows = (len(columns) - 1) // n_cols + 1 
    fig = plt.figure(figsize=(n_cols * 5, n_rows * 4)) 
 
    for i, col in enumerate(columns): 
        plt.subplot(n_rows, n_cols, i + 1)         
        counts, bin_edges = np.histogram(df[col].dropna(), bins=DEFAULT_DIST_BINS)
        plt.bar(bin_edges[:-1], counts, width=np.diff(bin_edges), align='edge', 
                color=COLOR_DIST_BLUE, edgecolor=COLOR_EDGE_BLACK, alpha=DEFAULT_ALPHA)
        
        plt.title(f'Distribution of {col}', fontsize=DEFAULT_TITLE_FONTSIZE, fontweight='bold') 
        plt.ylabel('Frequency') 
        plt.grid(axis='y', linestyle=GRID_LINESTYLE, alpha=GRID_ALPHA)
 
    plt.tight_layout() 
    plt.show()
    plt.close(fig) 
    gc.collect()

def plot_outliers(df, columns=None, n_cols=DEFAULT_OUTLIERS_N_COLS): 
    """Plots stunning boxplots. Fliers are turned OFF by default to prevent SVG freezing."""
    if columns is None: 
        columns = df.select_dtypes(include=[np.number]).columns.tolist() 
 
    if len(columns) == 0: return 
 
    n_rows = (len(columns) - 1) // n_cols + 1 
    fig = plt.figure(figsize=(n_cols * 4, n_rows * 3)) 
 
    for i, col in enumerate(columns): 
        plt.subplot(n_rows, n_cols, i + 1) 
        
        sns.boxplot(x=df[col], color=COLOR_OUTLIERS_GREEN, showfliers=True, width=BOXPLOT_WIDTH, linewidth=BOXPLOT_LINEWIDTH) 
        plt.title(f'Boxplot: {col}', fontsize=DEFAULT_TITLE_FONTSIZE, fontweight='bold') 
 
    plt.tight_layout() 
    plt.show()
    plt.close(fig)
    gc.collect()

def plot_categoricals(df, columns=None, n_cols=DEFAULT_CATEGORICALS_N_COLS): 
    """Plots bar charts for categorical data with automatic color mapping."""
    if columns is None: 
        columns = df.select_dtypes(include=['object', 'category']).columns.tolist() 
 
    if len(columns) == 0: return 
 
    n_rows = (len(columns) - 1) // n_cols + 1 
    fig = plt.figure(figsize=(n_cols * 5, n_rows * 4)) 
 
    for i, col in enumerate(columns): 
        plt.subplot(n_rows, n_cols, i + 1) 
        
        top_counts = df[col].value_counts().head(20)
        sns.barplot(x=top_counts.index, y=top_counts.values, palette=PALETTE_MAGMA, edgecolor=COLOR_EDGE_BLACK)
        
        plt.title(f'Top Counts: {col}', fontsize=DEFAULT_TITLE_FONTSIZE, fontweight='bold') 
        plt.xticks(rotation=LABEL_ROTATION_DEGREES)
        plt.ylabel('Frequency')
 
    plt.tight_layout() 
    plt.show()
    plt.close(fig)
    gc.collect()

def plot_target_distribution(df, target_col, task_type='classification'):
    """Visualizes the target variable to spot severe imbalances."""
    fig, ax = plt.subplots(figsize=TARGET_DIST_FIGSIZE)
    
    if task_type == 'classification':
        counts = df[target_col].value_counts()
        sns.barplot(x=counts.index, y=counts.values, palette=PALETTE_VIRIDIS, edgecolor=COLOR_EDGE_BLACK, ax=ax)
        ax.set_title(f'Target Imbalance Check: {target_col}', fontsize=HEADER_FONTSIZE, fontweight='bold')
        ax.set_ylabel('Count')
        for p in ax.patches:
            percentage = f'{100 * p.get_height() / len(df):.1f}%'
            ax.annotate(percentage, (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='bottom', fontsize=TEXT_FONTSIZE, fontweight='bold')
    else:
        sns.histplot(df[target_col], bins=DEFAULT_TARGET_BINS, kde=True, color=COLOR_REGRESSION_ORANGE, ax=ax)
        ax.set_title(f'Target Distribution: {target_col}', fontsize=HEADER_FONTSIZE, fontweight='bold')
        
    plt.tight_layout()
    plt.show()
    plt.close(fig)
    gc.collect()

def plot_correlation(df, target_col=None):
    """Plots Correlation Map with Massive sizing, 2-decimal annotations, explicit target highlights."""
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.empty: return

    corr = numeric_df.corr()
    fig = plt.figure(figsize=CORRELATION_FIGSIZE) 
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap=CMAP_SPECTRAL, 
                linewidths=CORR_LINEWIDTH, vmin=-1, vmax=1, annot_kws={"size": CORR_ANNOT_SIZE})
    
    plt.title('Correlation Matrix Heatmap', fontsize=CORR_TITLE_FONTSIZE, fontweight='bold')
    plt.tight_layout()
    plt.show()
    plt.close(fig)
    gc.collect()

    if target_col and target_col in numeric_df.columns:
        print(f"\n🎯 --- Top Correlations with Target: {target_col} ---")
        target_corr = corr[target_col].drop(target_col).sort_values(ascending=False)
        display(target_corr.head(CORR_HEAD_COUNT).to_frame(name='Positive Correlation').style.background_gradient(cmap=CMAP_GREENS))
        display(target_corr.tail(CORR_HEAD_COUNT).to_frame(name='Negative Correlation').style.background_gradient(cmap=CMAP_REDS_R))

def plot_dynamic_learning_curves(history):
    """
    Dynamically scans ANY Keras model's history and builds a custom dashboard.
    Works for single-task, multi-task, and custom-metric models automatically.
    """
    hist = history.history
    epochs = range(1, len(hist[list(hist.keys())[0]]) + 1)
    
    base_keys = [k for k in hist.keys() if not k.startswith('val_')]
    
    lr_keys = [k for k in base_keys if k in ['lr', 'learning_rate']]
    plot_keys = [k for k in base_keys if k not in lr_keys]
    
    if 'loss' in plot_keys:
        plot_keys.remove('loss')
        plot_keys.insert(0, 'loss')
    all_plots = plot_keys + lr_keys
    
    n_plots = len(all_plots)
    ncols = 2
    nrows = math.ceil(n_plots / ncols)
    
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, DASHBOARD_HEIGHT_FACTOR * nrows))
    fig.suptitle('Dynamic Model Training Dashboard', fontsize=DASHBOARD_TITLE_FONTSIZE, fontweight='bold', y=1.02)
    
    if n_plots <= 2:
        axes = axes.reshape(1, -1)
        
    axes_flat = axes.flatten()
    
    for i, key in enumerate(all_plots):
        ax = axes_flat[i]
        
        nice_title = key.replace('_', ' ').title()
        
        if key in lr_keys:
            ax.plot(epochs, hist[key], color=COLOR_LR_DECAY, lw=2)
            ax.set_title('Learning Rate Decay', fontsize=SUBPLOT_TITLE_FONTSIZE, fontweight='bold')
            ax.set_yscale('log')
            ax.set_xlabel('Epochs')
            ax.grid(True, linestyle=GRID_LINESTYLE, alpha=GRID_ALPHA)
            continue
            
        ax.plot(epochs, hist[key], color=COLOR_TRAIN_CURVE, linestyle='-', label=f'Train', lw=2)
        
        val_key = f'val_{key}'
        if val_key in hist:
            ax.plot(epochs, hist[val_key], color=COLOR_VAL_CURVE, linestyle='--', label=f'Validation', lw=2)
            
        ax.set_title(nice_title, fontsize=SUBPLOT_TITLE_FONTSIZE, fontweight='bold')
        ax.set_xlabel('Epochs')
        ax.legend(loc='best')
        ax.grid(True, linestyle=GRID_LINESTYLE, alpha=GRID_ALPHA)
        
    for j in range(len(all_plots), len(axes_flat)):
        axes_flat[j].axis('off')
        
    plt.tight_layout()
    plt.show()
