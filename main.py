import os
import warnings

# 1. Block Loky from searching for physical core execution binaries on Windows
os.environ["LOKY_MAX_CPU_COUNT"] = "4"  # Adjust to your local logical core preference if needed

# 2. Suppress standard library and third-party framework validation warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

from src.eda import run_eda
from src.train import main_training_pipeline

def main():
    print("Starting Credit Scoring Production Pipeline Lifecycle...")
    
    # Run Diagnostic Profiling
    run_eda()
    
    # Run Baseline -> Tuning -> SHAP Framework
    main_training_pipeline()
    
    print("\nExecution Finished. Production artifacts successfully populated.")

if __name__ == "__main__":
    main()