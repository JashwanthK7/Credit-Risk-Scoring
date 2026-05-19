import os
import warnings

from src.eda import run_eda
from src.train import main_training_pipeline

os.environ['LOKY_MAX_CPU_COUNT'] = str(os.cpu_count() or 4)

warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

def main():
    print('Starting Credit Scoring Production Pipeline Lifecycle...')
    run_eda()
    main_training_pipeline()
    print('\nExecution Finished. Production artifacts successfully populated.')

if __name__ == '__main__':
    main()