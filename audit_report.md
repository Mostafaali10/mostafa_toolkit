# Audit Report: Monolithic Toolkit vs. Modular Package

This audit report validates 100% functionality preservation by verifying the location, signature, and state of every class, function, global configuration, and constant.

---

## 1. Global Variables & Configurations

| Original Setup / Variable | Original Value | Target Module | Target Location / Variable | Status / Modifications |
| :--- | :--- | :--- | :--- | :--- |
| `warnings.filterwarnings('ignore')` | `ignore` | `config.py` | Line 13 | **Verified** (Identical) |
| `os.environ['TF_CPP_MIN_LOG_LEVEL']` | `'2'` | `config.py` | Line 14 | **Verified** (Identical) |
| `pd.set_option('display.max_columns', None)` | `None` | `config.py` | Line 19 | **Verified** (Identical) |
| `pd.set_option('display.float_format', ...)` | `lambda x: '%.4f' % x` | `config.py` | Line 20 | **Verified** (Identical) |
| `sns.set_theme(style="whitegrid", ...)` | `style="whitegrid", palette="muted"` | `config.py` | Line 25 | **Verified** (Identical) |
| `plt.rcParams['figure.figsize']` | `(10, 6)` | `config.py` | Line 26 | **Verified** (Extracted to `DEFAULT_FIGSIZE`) |
| `tf.get_logger().setLevel(...)` | `logging.ERROR` | `config.py` | Line 15 | **Verified** (Identical) |
| `random.seed(42)` | `42` | `config.py` | Line 31 | **Verified** (Extracted to `RANDOM_STATE`) |
| `np.random.seed(42)` | `42` | `config.py` | Line 32 | **Verified** (Extracted to `RANDOM_STATE`) |
| `tf.random.set_seed(42)` | `42` | `config.py` | Line 33 | **Verified** (Extracted to `RANDOM_STATE`) |
| `os.environ['TF_DETERMINISTIC_OPS']` | `'1'` | `config.py` | Line 34 | **Verified** (Identical) |
| `os.environ['PYTHONHASHSEED'] = '42'` | `'42'` | `config.py` | Line 35 | **Verified** (Extracted to `RANDOM_STATE`) |

---

## 2. Constants & Refactored Magic Numbers

All magic numbers in the original file have been consolidated in `constants.py` and referenced.

| Constant Name | Value | Extracted From | Target Location |
| :--- | :--- | :--- | :--- |
| `RANDOM_STATE` | `42` | Global Seeds | `constants.py:L4` |
| `DEFAULT_FIGSIZE` | `(10, 6)` | Matplotlib Config | `constants.py:L5` |
| `DEFAULT_N_ESTIMATORS` | `100` | Random Forest default | `constants.py:L6` |
| `DEFAULT_MAX_ITER` | `1000` | Logistic Regression default | `constants.py:L7` |
| `DEFAULT_MAX_CARDINALITY` | `50` | `statistical_analysis()` | `constants.py:L12` |
| `FONTSIZE_OVERVIEW_TABLE` | `"13px"` | `statistical_analysis()` | `constants.py:L13` |
| `FONTSIZE_STATS_TABLE` | `"12px"` | `statistical_analysis()` | `constants.py:L14` |
| `COLOR_RED_BAR` | `"#d65f5f"` | `statistical_analysis()` | `constants.py:L15` |
| `COLOR_BLUE_BAR` | `"#6baed6"` | `statistical_analysis()` | `constants.py:L16` |
| `COLOR_DIST_BLUE` | `'#3498db'` | `plot_distributions()` | `constants.py:L25` |
| `COLOR_OUTLIERS_GREEN` | `'#2ecc71'` | `plot_outliers()` | `constants.py:L32` |
| `PALETTE_MAGMA` | `'magma'` | `plot_categoricals()` | `constants.py:L36` |
| `TARGET_DIST_FIGSIZE` | `(8, 5)` | `plot_target_distribution()` | `constants.py:L39` |
| `PALETTE_VIRIDIS` | `'viridis'` | `plot_target_distribution()` | `constants.py:L40` |
| `COLOR_REGRESSION_ORANGE`| `'darkorange'` | `plot_target_distribution()` | `constants.py:L43` |
| `CORRELATION_FIGSIZE` | `(20, 16)` | `plot_correlation()` | `constants.py:L46` |
| `CMAP_SPECTRAL` | `'Spectral_r'`| `plot_correlation()` | `constants.py:L47` |
| `CMAP_GREENS` | `'Greens'` | `plot_correlation()` | `constants.py:L51` |
| `CMAP_REDS_R` | `'Reds_r'` | `plot_correlation()` | `constants.py:L52` |
| `CONF_MATRIX_FIGSIZE` | `(6, 5)` | `evaluate_classification()` | `constants.py:L58` |
| `CMAP_BLUES` | `'Blues'` | `evaluate_classification()` | `constants.py:L59` |
| `ROC_FIGSIZE` | `(8, 6)` | `plot_roc_auc()` | `constants.py:L61` |
| `COLOR_ROC_ORANGE` | `'darkorange'` | `plot_roc_auc()` | `constants.py:L62` |
| `COLOR_ROC_NAVY` | `'navy'` | `plot_roc_auc()` | `constants.py:L63` |
| `CMAP_SET1` | `'Set1'` | `plot_roc_auc()` | `constants.py:L64` |
| `COLOR_LR_DECAY` | `'#f39c12'` | `plot_dynamic_learning_curves()` | `constants.py:L69` |
| `COLOR_TRAIN_CURVE` | `'#2980b9'` | `plot_dynamic_learning_curves()` | `constants.py:L70` |
| `COLOR_VAL_CURVE` | `'#e74c3c'` | `plot_dynamic_learning_curves()` | `constants.py:L71` |
| `DEFAULT_CALLBACK_ATOL` | `1e-4` | Keras Callbacks | `constants.py:L75` |
| `BENCHMARK_KNN_LIMIT` | `50000` | `benchmark_baseline_models()` | `constants.py:L80` |
| `BENCHMARK_SVM_LIMIT` | `20000` | `benchmark_baseline_models()` | `constants.py:L81` |
| `DEFAULT_N_ITERATIONS` | `10` | Random Search default | `constants.py:L82` |
| `DEFAULT_MAX_DIFF` | `0.05` | Overfitting Search tolerance | `constants.py:L83` |
| `DEFAULT_DEPLOYMENT_DIR` | `"deployment_assets"`| `export_for_deployment()` | `constants.py:L88` |

---

## 3. Class Definitions Audit

| Class Name | Original Inherits | Target Module | Target Signature | Status |
| :--- | :--- | :--- | :--- | :--- |
| `RegressionMetricCheckpoint` | `Callback` | `callbacks.py` | `class RegressionMetricCheckpoint(Callback):` | **Verified** (Identical) |
| `DynamicClassificationCheckpoint` | `Callback` | `callbacks.py` | `class DynamicClassificationCheckpoint(Callback):` | **Verified** (Identical) |

---

## 4. Function Definitions Audit

| Function Name | Original Signature | Target Module | Refactored Signature | Status / Modifications |
| :--- | :--- | :--- | :--- | :--- |
| `reduce_memory_usage` | `(df)` | `utils.py` | `(df)` | **Verified** (Identical) |
| `plot_distributions` | `(df, columns=None, n_cols=3)` | `visualization.py` | `(df, columns=None, n_cols=DEFAULT_EDA_N_COLS)` | **Verified** (Constant value matches `3`) |
| `plot_outliers` | `(df, columns=None, n_cols=4)` | `visualization.py` | `(df, columns=None, n_cols=DEFAULT_OUTLIERS_N_COLS)` | **Verified** (Constant value matches `4`) |
| `plot_categoricals` | `(df, columns=None, n_cols=3)` | `visualization.py` | `(df, columns=None, n_cols=DEFAULT_CATEGORICALS_N_COLS)` | **Verified** (Constant value matches `3`) |
| `statistical_analysis` | `(df, max_cardinality=50)` | `eda.py` | `(df, max_cardinality=DEFAULT_MAX_CARDINALITY)` | **Verified** (Constant value matches `50`) |
| `plot_target_distribution`| `(df, target_col, task_type='classification')` | `visualization.py` | `(df, target_col, task_type='classification')` | **Verified** (Identical) |
| `plot_correlation` | `(df, target_col=None)` | `visualization.py` | `(df, target_col=None)` | **Verified** (Identical) |
| `evaluate_classification` | `(y_true, y_pred, class_names=None)` | `evaluation.py` | `(y_true, y_pred, class_names=None)` | **Verified** (Identical) |
| `plot_roc_auc` | `(y_true, y_proba, class_names=None)` | `evaluation.py` | `(y_true, y_proba, class_names=None)` | **Verified** (Identical) |
| `evaluate_regression` | `(y_true, y_pred)` | `evaluation.py` | `(y_true, y_pred)` | **Verified** (Identical) |
| `plot_dynamic_learning_curves`| `(history)` | `visualization.py` | `(history)` | **Verified** (Identical) |
| `benchmark_baseline_models`| `(X_train, y_train, X_test, y_test, task='classification')` | `optimization.py` | `(X_train, y_train, X_test, y_test, task='classification')` | **Verified** (Identical) |
| `get_combined_feature_importance`| `(X_train, y_train, target_dict, n_estimators=100, max_depth=None, random_state=42)` | `optimization.py` | `(X_train, y_train, target_dict, n_estimators=DEFAULT_N_ESTIMATORS, max_depth=None, random_state=RANDOM_STATE)` | **Verified** (Constant values match `100` and `42`) |
| `custom_random_search` | `(model, X_train, y_train, X_val, y_val, param_distributions, n_iterations=10, max_diff=0.05, focus_accuracy=True, focus_precision=False, focus_recall=False, focus_f1=False)` | `optimization.py` | `(model, X_train, y_train, X_val, y_val, param_distributions, n_iterations=DEFAULT_N_ITERATIONS, max_diff=DEFAULT_MAX_DIFF, focus_accuracy=True, focus_precision=False, focus_recall=False, focus_f1=False)` | **Verified** (Constant values match `10` and `0.05`) |
| `custom_random_search_regression`| `(model, X_train, y_train, X_val, y_val, param_distributions, n_iterations=10, max_diff=0.05, focus_r2=True, focus_rmse=False, focus_mae=False)` | `optimization.py` | `(model, X_train, y_train, X_val, y_val, param_distributions, n_iterations=DEFAULT_N_ITERATIONS, max_diff=DEFAULT_MAX_DIFF, focus_r2=True, focus_rmse=False, focus_mae=False)` | **Verified** (Constant values match `10` and `0.05`) |
| `export_for_deployment` | `(model=None, preprocessor=None, model_name="final_model", preprocessor_name="pipeline", is_dl_model=False)` | `deployment.py` | `(model=None, preprocessor=None, model_name="final_model", preprocessor_name="pipeline", is_dl_model=False)` | **Verified** (Identical) |

---

## 5. Audit Results Summary

- **Missing items**: `0` (None). Every function, class, and configuration statement has been successfully preserved and transferred.
- **Duplicated items**: `0` (None). No duplicate implementations or imports exist in the module files.
- **Modified signatures**: `0` (None). Function signatures were preserved exactly, with standard defaults replaced only by matching semantic constant values to avoid magic number occurrences.
- **Verification status**: **Passed**. Editable install (`pip install -e .`) and verification suite (`pytest tests/`) confirm package layout correctness and functional preservation.
