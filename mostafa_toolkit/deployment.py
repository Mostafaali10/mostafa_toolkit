import os
import joblib

from .constants import DEFAULT_DEPLOYMENT_DIR

def export_for_deployment(model=None, preprocessor=None, model_name="final_model", preprocessor_name="pipeline", is_dl_model=False):
    """
    Safely exports models and preprocessing artifacts (Scalers, PCAs, Encoders) 
    for production deployment.
    """
    print("="*50)
    print("📦 INITIATING DEPLOYMENT EXPORT 📦")
    print("="*50)
    
    export_dir = DEFAULT_DEPLOYMENT_DIR
    if not os.path.exists(export_dir):
        os.makedirs(export_dir)
        print(f"📁 Created directory: ./{export_dir}/")
        
    if preprocessor is not None:
        prep_path = f"{export_dir}/{preprocessor_name}.joblib"
        joblib.dump(preprocessor, prep_path)
        print(f"✅ Preprocessor saved   : {prep_path}")
        print(f"   (Includes all Scalers, Encoders, and Dim Reduction states)")
    
    if model is not None:
        if is_dl_model:
            model_path = f"{export_dir}/{model_name}.keras"
            model.save(model_path)
            print(f"✅ Deep Learning saved  : {model_path}")
        else:
            model_path = f"{export_dir}/{model_name}.joblib"
            joblib.dump(model, model_path)
            print(f"✅ Machine Learning saved: {model_path}")
            
    print("="*50)
