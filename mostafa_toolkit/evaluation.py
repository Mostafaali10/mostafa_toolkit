import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    auc,
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from .constants import (
    CONF_MATRIX_FIGSIZE,
    CMAP_BLUES,
    ROC_FIGSIZE,
    COLOR_ROC_ORANGE,
    COLOR_ROC_NAVY,
    CMAP_SET1
)

def evaluate_classification(y_true, y_pred, class_names=None):
    """Prints all classification metrics and plots an elegant Confusion Matrix."""
    print("="*55)
    print("📊 CLASSIFICATION REPORT")
    print("="*55)
    print(classification_report(y_true, y_pred, target_names=class_names))
    
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    print("-" * 55)
    print(f"Accuracy           : {acc:.4f}")
    print(f"Weighted Precision : {prec:.4f}")
    print(f"Weighted Recall    : {rec:.4f}")
    print(f"Weighted F1-Score  : {f1:.4f}")
    print("="*55)
    
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=CONF_MATRIX_FIGSIZE)
    sns.heatmap(cm, annot=True, fmt='d', cmap=CMAP_BLUES, cbar=False, 
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
    ax.set_ylabel('Actual Label')
    ax.set_xlabel('Predicted Label')
    plt.show()
    plt.close(fig)

def plot_roc_auc(y_true, y_proba, class_names=None):
    """Plots a stunning ROC Curve for both Binary and Multiclass Classification."""
    from sklearn.preprocessing import label_binarize
    
    fig, ax = plt.subplots(figsize=ROC_FIGSIZE)
    
    if len(y_proba.shape) > 1 and y_proba.shape[1] > 2:
        n_classes = y_proba.shape[1]
        classes = np.unique(y_true)
        y_true_bin = label_binarize(y_true, classes=classes)
        colors = plt.cm.get_cmap(CMAP_SET1, n_classes)
        
        for i in range(n_classes):
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_proba[:, i])
            roc_auc = auc(fpr, tpr)
            c_name = class_names[i] if class_names else f"Class {classes[i]}"
            ax.plot(fpr, tpr, color=colors(i), lw=2, label=f'{c_name} (AUC = {roc_auc:.3f})')
            
        ax.set_title('Multiclass ROC Curve (One-vs-Rest)', fontsize=14, fontweight='bold')
        
    else:
        if len(y_proba.shape) > 1 and y_proba.shape[1] == 2:
            y_proba = y_proba[:, 1]
            
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        roc_auc = auc(fpr, tpr)
        
        ax.plot(fpr, tpr, color=COLOR_ROC_ORANGE, lw=2, label=f'ROC Curve (AUC = {roc_auc:.3f})')
        ax.set_title('Receiver Operating Characteristic (ROC)', fontsize=14, fontweight='bold')
        
    ax.plot([0, 1], [0, 1], color=COLOR_ROC_NAVY, lw=2, linestyle='--', label='Random Guess')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.legend(loc="lower right")
    plt.grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.show()
    plt.close(fig)

def evaluate_regression(y_true, y_pred):
    """Calculates all key regression metrics and plots True vs Predicted."""
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    
    print("="*55)
    print("📈 REGRESSION METRICS")
    print("="*55)
    print(f"MAE  (Mean Absolute Error)     : {mae:.4f}")
    print(f"MSE  (Mean Squared Error)      : {mse:.4f}")
    print(f"RMSE (Root Mean Squared Error) : {rmse:.4f}")
    print(f"R²   (Variance Explained)      : {r2:.4f}")
    print("="*55)
