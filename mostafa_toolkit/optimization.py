import copy
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display
from sklearn.model_selection import ParameterSampler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from xgboost import XGBClassifier, XGBRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    r2_score,
    mean_absolute_error,
    mean_squared_error
)

from .constants import (
    RANDOM_STATE,
    DEFAULT_N_ESTIMATORS,
    DEFAULT_MAX_ITER,
    BENCHMARK_KNN_LIMIT,
    BENCHMARK_SVM_LIMIT,
    DEFAULT_N_ITERATIONS,
    DEFAULT_MAX_DIFF
)

def benchmark_baseline_models(X_train, y_train, X_test, y_test, task='classification'):
    """
    Trains and evaluates a massive suite of baseline models, 
    returning a comprehensive, sorted diagnostic leaderboard.
    """
    print(f"🚀 Benchmarking {task.upper()} Models... This might take a moment!")
    total_samples = len(X_train) + len(X_test)
    
    if task == 'classification':
        # 1. Base models that ALWAYS run (They scale well)
        models = {
            'Logistic Regression': LogisticRegression(max_iter=DEFAULT_MAX_ITER, random_state=RANDOM_STATE),
            'Decision Tree': DecisionTreeClassifier(random_state=RANDOM_STATE),
            'Random Forest': RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1),
            'Gradient Boosting': GradientBoostingClassifier(random_state=RANDOM_STATE),
            'XGBoost': XGBClassifier(random_state=RANDOM_STATE, use_label_encoder=False, eval_metric='logloss', n_jobs=-1)
        }
        
        # 2. Add KNN if under 50k samples
        if total_samples <= BENCHMARK_KNN_LIMIT:
            models['KNN'] = KNeighborsClassifier(n_jobs=-1)
            
        # 3. Add SVM if under 20k samples (Strict limit to prevent freezing)
        if total_samples <= BENCHMARK_SVM_LIMIT:
            models['SVC (Support Vector)'] = SVC(random_state=RANDOM_STATE)
            
    else:
        # 1. Base models that ALWAYS run
        models = {
            'Linear Regression': LinearRegression(),
            'Decision Tree': DecisionTreeRegressor(random_state=RANDOM_STATE),
            'Random Forest': RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1),
            'Gradient Boosting': GradientBoostingRegressor(random_state=RANDOM_STATE),
            'XGBoost': XGBRegressor(random_state=RANDOM_STATE, n_jobs=-1)
        }
        
        # 2. Add KNN if under 50k samples
        if total_samples <= BENCHMARK_KNN_LIMIT:
            models['KNN'] = KNeighborsRegressor(n_jobs=-1)
            
        # 3. Add SVM if under 20k samples
        if total_samples <= BENCHMARK_SVM_LIMIT:
            models['SVR (Support Vector)'] = SVR()
            
    results = []
    
    for name, model in models.items():
        print(f"Training {name}...")
        start_time = time.time()
        
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        
        train_time = time.time() - start_time
        
        if task == 'classification':
            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
            prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            
            results.append({
                'Model': name, 
                'Accuracy': acc, 
                'F1-Score': f1, 
                'Precision': prec,
                'Recall': rec,
                'Time (s)': round(train_time, 2)
            })
        else:
            r2 = r2_score(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            
            results.append({
                'Model': name, 
                'R2-Score': r2, 
                'MAE': mae,
                'MSE': mse,
                'RMSE': rmse, 
                'Time (s)': round(train_time, 2)
            })
            
    results_df = pd.DataFrame(results)
    
    if task == 'classification':
        results_df = results_df.sort_values(by='F1-Score', ascending=False).reset_index(drop=True)
    else:
        results_df = results_df.sort_values(by='R2-Score', ascending=False).reset_index(drop=True)
        
    print("\n🏆 --- ULTIMATE BASELINE LEADERBOARD --- 🏆")
    display(results_df.style.background_gradient(cmap='Blues'))

def get_combined_feature_importance(X_train, y_train, target_dict, n_estimators=DEFAULT_N_ESTIMATORS, max_depth=None, random_state=RANDOM_STATE):
    """
    Trains models on ANY number of targets, extracts their feature importances, 
    and returns a combined summary DataFrame.
    
    Parameters:
    - X_train: DataFrame of input features.
    - y_train: DataFrame containing the target columns.
    - target_dict: A dictionary mapping the target column names to their problem type.
                   Example: {'cardio': 'classification', 'blood_pressure': 'regression'}
    - n_estimators: Number of trees in the random forests.
    - max_depth: Maximum depth of the trees (default=None).
    - random_state: Seed for reproducibility.
    
    Returns:
    - feature_importance_df: A sorted DataFrame containing the individual and combined importances.
    """
    
    print(f"Training models on {len(target_dict)} target(s)...")
    
    # 1. Initialize the base DataFrame with feature names
    feature_importance_df = pd.DataFrame({'Feature': X_train.columns})
    
    # Keep track of the target column names for averaging later
    target_names = []
    
    # 2. Dynamically loop through every target you provided
    for target_col, task_type in target_dict.items():
        print(f" -> Evaluating '{target_col}' as {task_type.upper()}...")
        
        # Choose the right algorithm based on the task type
        if task_type.lower() == 'classification':
            rf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, n_jobs=-1, random_state=random_state)
        elif task_type.lower() == 'regression':
            rf = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, n_jobs=-1, random_state=random_state)
        else:
            raise ValueError(f"Task type must be 'classification' or 'regression'. You passed: {task_type}")
            
        # Train the model and extract importance
        rf.fit(X_train, y_train[target_col])
        
        # Add a new column to our DataFrame with this specific target's results
        feature_importance_df[target_col] = rf.feature_importances_
        target_names.append(target_col)

    # 3. Calculate the Combined Score (Average across all evaluated targets)
    feature_importance_df['Combined_Importance'] = feature_importance_df[target_names].mean(axis=1)
    feature_importance_df = feature_importance_df.sort_values(by='Combined_Importance', ascending=False)

    # 4. Plot the Combined Results
    plt.figure(figsize=(14, 10))
    sns.barplot(x='Combined_Importance', y='Feature', data=feature_importance_df, palette='magma')
    plt.title(f'Combined Feature Importance Across {len(target_dict)} Targets', fontsize=16)
    plt.xlabel('Average Relative Importance', fontsize=12)
    plt.ylabel('Feature', fontsize=12)
    plt.tight_layout()
    plt.show()

def custom_random_search(
    model, 
    X_train, y_train, 
    X_val, y_val, 
    param_distributions, 
    n_iterations=DEFAULT_N_ITERATIONS, 
    max_diff=DEFAULT_MAX_DIFF, 
    focus_accuracy=True, 
    focus_precision=False, 
    focus_recall=False, 
    focus_f1=False
):
    """
    Performs a custom randomized search to find the best hyperparameters while strictly 
    controlling for overfitting based on a targeted metric's Train/Validation difference.
    """
    
    # Ensure at least one focus metric is selected
    if not any([focus_accuracy, focus_precision, focus_recall, focus_f1]):
        raise ValueError("You must set at least one focus metric (e.g., focus_accuracy=True)")

    # 1. Create a random sampler for the hyperparameters
    sampler = ParameterSampler(param_distributions, n_iter=n_iterations, random_state=RANDOM_STATE)
    
    best_model = None
    best_val_score = -1
    best_params = None
    
    # 2. Setup the visual table for printing
    print(f"{'Iter':<5} | {'Hyperparameters':<45} | {'Set':<5} | {'Acc':<6} | {'Prec':<6} | {'Rec':<6} | {'F1':<6} | {'Target Diff'}")
    print("-" * 105)
    
    for i, params in enumerate(sampler):
        # Apply parameters to the model
        current_model = copy.deepcopy(model)
        current_model.set_params(**params)
        
        # Train the model
        current_model.fit(X_train, y_train)
        
        # Predict on both sets
        y_train_pred = current_model.predict(X_train)
        y_val_pred = current_model.predict(X_val)
        
        # Calculate metrics (assuming binary classification)
        def get_metrics(y_true, y_pred):
            return {
                'accuracy': accuracy_score(y_true, y_pred),
                'precision': precision_score(y_true, y_pred, zero_division=0),
                'recall': recall_score(y_true, y_pred, zero_division=0),
                'f1': f1_score(y_true, y_pred, zero_division=0)
            }
            
        train_metrics = get_metrics(y_train, y_train_pred)
        val_metrics = get_metrics(y_val, y_val_pred)
        
        # Calculate the "Target Score" based on user's boolean flags
        train_focus_scores = []
        val_focus_scores = []
        
        if focus_accuracy:
            train_focus_scores.append(train_metrics['accuracy'])
            val_focus_scores.append(val_metrics['accuracy'])
        if focus_precision:
            train_focus_scores.append(train_metrics['precision'])
            val_focus_scores.append(val_metrics['precision'])
        if focus_recall:
            train_focus_scores.append(train_metrics['recall'])
            val_focus_scores.append(val_metrics['recall'])
        if focus_f1:
            train_focus_scores.append(train_metrics['f1'])
            val_focus_scores.append(val_metrics['f1'])
            
        # Average the focused metrics (in case user selected more than one)
        train_target = np.mean(train_focus_scores)
        val_target = np.mean(val_focus_scores)
        
        # Calculate the absolute difference to check for overfitting
        diff = abs(train_target - val_target)
        
        # Format params string so it doesn't break the table
        param_str = str(params)
        if len(param_str) > 42:
            param_str = param_str[:39] + "..."
            
        # Print the Train Row
        print(f"{i+1:<5} | {param_str:<45} | {'Train':<5} | "
              f"{train_metrics['accuracy']:.4f} | {train_metrics['precision']:.4f} | "
              f"{train_metrics['recall']:.4f} | {train_metrics['f1']:.4f} | {diff:.4f}")
              
        # Print the Validation Row
        print(f"{'':<5} | {'':<45} | {'Val':<5} | "
              f"{val_metrics['accuracy']:.4f} | {val_metrics['precision']:.4f} | "
              f"{val_metrics['recall']:.4f} | {val_metrics['f1']:.4f} | {'':<6}")
        print("-" * 105)
        
        # 3. Best Model Selection Logic
        # Is the difference less than the max_diff threshold?
        if diff <= max_diff:
            # Does it have the highest validation score for the targeted metric so far?
            if val_target > best_val_score:
                best_val_score = val_target
                best_model = copy.deepcopy(current_model)
                best_params = params

    # 4. Final Summary
    print("\n" + "="*50)
    if best_model is not None:
        print("🏆 BEST MODEL FOUND 🏆")
        print(f"Hyperparameters: {best_params}")
        print(f"Target Validation Score: {best_val_score:.4f}")
        print(f"Difference gap was within the {max_diff} limit.")
    else:
        print("⚠️ NO MODEL MET THE CRITERIA ⚠️")
        print(f"None of the {n_iterations} iterations resulted in a Train/Val difference <= {max_diff}.")
        print("Try increasing 'max_diff' or adjusting your hyperparameter distributions to be more restrictive (e.g., shallower trees).")
    print("="*50)
    
    return best_model

def custom_random_search_regression(
    model, 
    X_train, y_train, 
    X_val, y_val, 
    param_distributions, 
    n_iterations=DEFAULT_N_ITERATIONS, 
    max_diff=DEFAULT_MAX_DIFF, 
    focus_r2=True,     
    focus_rmse=False, 
    focus_mae=False
):
    """
    Performs a custom randomized search for REGRESSION models. 
    Strictly controls for overfitting based on the Train/Validation difference.
    """
    
    # 1. Enforce that EXACTLY one focus metric is selected
    focus_flags = [focus_r2, focus_rmse, focus_mae]
    if sum(focus_flags) != 1:
        raise ValueError("For regression, you must select EXACTLY ONE focus metric (e.g., only focus_r2=True).")

    # 2. Determine the optimization direction (minimize errors, maximize R2)
    if focus_r2:
        best_val_score = float('-inf')  # We want the highest R2
        higher_is_better = True
        target_name = "R2"
    else:
        best_val_score = float('inf')   # We want the lowest Error (RMSE/MAE)
        higher_is_better = False
        target_name = "RMSE" if focus_rmse else "MAE"

    # Create a random sampler for the hyperparameters
    sampler = ParameterSampler(param_distributions, n_iter=n_iterations, random_state=RANDOM_STATE)
    best_model = None
    best_params = None
    
    # 3. Setup the visual table for printing
    print(f"{'Iter':<5} | {'Hyperparameters':<45} | {'Set':<5} | {'R²':<8} | {'RMSE':<8} | {'MAE':<8} | {target_name+' Diff'}")
    print("-" * 105)
    
    for i, params in enumerate(sampler):
        # Apply parameters to the model
        current_model = copy.deepcopy(model)
        current_model.set_params(**params)
        
        # Train the model
        current_model.fit(X_train, y_train)
        
        # Predict on both sets
        y_train_pred = current_model.predict(X_train)
        y_val_pred = current_model.predict(X_val)
        
        # Calculate Regression metrics
        def get_metrics(y_true, y_pred):
            return {
                'r2': r2_score(y_true, y_pred),
                'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
                'mae': mean_absolute_error(y_true, y_pred)
            }
            
        train_metrics = get_metrics(y_train, y_train_pred)
        val_metrics = get_metrics(y_val, y_val_pred)
        
        # Extract the target scores based on user selection
        if focus_r2:
            train_target, val_target = train_metrics['r2'], val_metrics['r2']
        elif focus_rmse:
            train_target, val_target = train_metrics['rmse'], val_metrics['rmse']
        elif focus_mae:
            train_target, val_target = train_metrics['mae'], val_metrics['mae']
            
        # Calculate the absolute difference to check for overfitting
        diff = abs(train_target - val_target)
        
        # Format params string so it doesn't break the table
        param_str = str(params)
        if len(param_str) > 42:
            param_str = param_str[:39] + "..."
            
        # Print the Train Row
        print(f"{i+1:<5} | {param_str:<45} | {'Train':<5} | "
              f"{train_metrics['r2']:<8.4f} | {train_metrics['rmse']:<8.4f} | "
              f"{train_metrics['mae']:<8.4f} | {diff:.4f}")
              
        # Print the Validation Row
        print(f"{'':<5} | {'':<45} | {'Val':<5} | "
              f"{val_metrics['r2']:<8.4f} | {val_metrics['rmse']:<8.4f} | "
              f"{val_metrics['mae']:<8.4f} | {'':<6}")
        print("-" * 105)
        
        # 4. Best Model Selection Logic
        if diff <= max_diff:
            # Check if this model is better than our previous best
            is_better = (val_target > best_val_score) if higher_is_better else (val_target < best_val_score)
            
            if is_better:
                best_val_score = val_target
                best_model = copy.deepcopy(current_model)
                best_params = params

    # 5. Final Summary
    print("\n" + "="*50)
    if best_model is not None:
        print("🏆 BEST REGRESSION MODEL FOUND 🏆")
        print(f"Hyperparameters: {best_params}")
        print(f"Target Validation Score ({target_name}): {best_val_score:.4f}")
        print(f"Train/Val Difference was within the {max_diff} limit.")
    else:
        print("⚠️ NO MODEL MET THE CRITERIA ⚠️")
        print(f"None of the iterations resulted in a {target_name} difference <= {max_diff}.")
        print("Try increasing 'max_diff' or restricting your hyperparameters.")
    print("="*50)
    
    return best_model
