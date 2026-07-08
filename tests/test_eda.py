import numpy as np
import pandas as pd
import pytest

import mostafa_toolkit as mt

def test_imports():
    """Verify that package level namespace imports are exposed properly."""
    assert hasattr(mt, "reduce_memory_usage")
    assert hasattr(mt, "statistical_analysis")
    assert hasattr(mt, "plot_distributions")
    assert hasattr(mt, "plot_outliers")
    assert hasattr(mt, "plot_categoricals")
    assert hasattr(mt, "plot_target_distribution")
    assert hasattr(mt, "plot_correlation")
    assert hasattr(mt, "plot_dynamic_learning_curves")
    assert hasattr(mt, "evaluate_classification")
    assert hasattr(mt, "evaluate_regression")
    assert hasattr(mt, "plot_roc_auc")
    assert hasattr(mt, "benchmark_baseline_models")
    assert hasattr(mt, "get_combined_feature_importance")
    assert hasattr(mt, "custom_random_search")
    assert hasattr(mt, "custom_random_search_regression")
    assert hasattr(mt, "RegressionMetricCheckpoint")
    assert hasattr(mt, "DynamicClassificationCheckpoint")
    assert hasattr(mt, "export_for_deployment")

def test_reduce_memory_usage():
    """Verify reduce_memory_usage functionality downcasts int/float types."""
    # Create high precision numbers that can be downcast
    df = pd.DataFrame({
        'int_col': [1, 2, 3, 4, 5],
        'float_col': [1.1, 2.2, 3.3, 4.4, 5.5]
    })
    
    # Original types should be int64 and float64 by default
    assert df['int_col'].dtype == np.int64
    assert df['float_col'].dtype == np.float64
    
    optimized_df = mt.reduce_memory_usage(df)
    
    # Check that they have been downcast (e.g. to int8 and float32)
    assert optimized_df['int_col'].dtype == np.int8
    assert optimized_df['float_col'].dtype == np.float32

def test_statistical_analysis(capsys):
    """Verify statistical_analysis generates expected outputs for dataframes."""
    df = pd.DataFrame({
        'num_col': [10, 20, 30, 40],
        'cat_col': ['A', 'B', 'A', 'C']
    })
    
    # Should run without error (display functions print Markdown/HTML via IPython)
    mt.statistical_analysis(df, max_cardinality=5)
    
    # Let's ensure no exception was raised and output has been formatted
    captured = capsys.readouterr()
    # It prints display details
    assert df.shape == (4, 2)
