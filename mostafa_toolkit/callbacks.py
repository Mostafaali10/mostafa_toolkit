import numpy as np
from tensorflow.keras.callbacks import Callback

from .constants import DEFAULT_CALLBACK_ATOL

class RegressionMetricCheckpoint(Callback): 
    def __init__(self, filepath, monitor_metric="val_mse", mode="min"): 
        super().__init__() 
        self.filepath = filepath 
        self.monitor_metric = monitor_metric 
        self.mode = mode.lower() # Accepts 'min' or 'max'
        
        # Start at infinity if minimizing (MSE/MAE), or negative infinity if maximizing (R2)
        if self.mode == "min":
            self.best_metric = np.inf 
        elif self.mode == "max":
            self.best_metric = -np.inf
        else:
            raise ValueError("Mode must be 'min' or 'max'")
            
        self.best_loss = np.inf 
  
    def on_epoch_end(self, epoch, logs=None): 
        current_metric = logs.get(self.monitor_metric) 
        val_loss = logs.get("val_loss") 
  
        if current_metric is None:
            print(f"\nWarning: '{self.monitor_metric}' not found in logs.")
            return

        # Check if the primary metric improved based on the chosen mode
        improved = False
        if self.mode == "min":
            improved = current_metric < self.best_metric
        elif self.mode == "max":
            improved = current_metric > self.best_metric

        # Tie-breaker: If the primary metric is effectively identical, check the loss
        is_tied = np.isclose(current_metric, self.best_metric, atol=DEFAULT_CALLBACK_ATOL)

        if improved or (is_tied and val_loss < self.best_loss): 
              
            print(f"\nEpoch {epoch+1}: {self.monitor_metric}={current_metric:.4f}, val_loss={val_loss:.4f} " 
                  f"--> saving best model") 
              
            self.best_metric = current_metric 
            self.best_loss = val_loss 
            self.model.save(self.filepath) 


class DynamicClassificationCheckpoint(Callback): 
    def __init__(self, filepath, monitor_metric="val_accuracy"): 
        super().__init__() 
        self.filepath = filepath 
        self.monitor_metric = monitor_metric  # e.g., 'val_custom_f1', 'val_precision'
        
        # We start at negative infinity because we want these metrics to INCREASE
        self.best_metric = -np.inf 
        self.best_loss = np.inf 
  
    def on_epoch_end(self, epoch, logs=None): 
        # Safely grab the target metric from the logs
        current_metric = logs.get(self.monitor_metric) 
        val_loss = logs.get("val_loss") 
  
        # Safety check: ensure the metric actually exists in the logs
        if current_metric is None:
            print(f"\nWarning: '{self.monitor_metric}' not found in logs. Cannot save.")
            return

        # If primary metric improves OR (tied metric AND lower loss)
        if (current_metric > self.best_metric) or (np.isclose(current_metric, self.best_metric, atol=DEFAULT_CALLBACK_ATOL) and val_loss < self.best_loss): 
              
            print(f"\nEpoch {epoch+1}: {self.monitor_metric}={current_metric:.4f}, val_loss={val_loss:.4f} " 
                  f"--> saving best model") 
              
            self.best_metric = current_metric 
            self.best_loss = val_loss 
            self.model.save(self.filepath)
