import os
import zipfile
from kaggle.api.kaggle_api_extended import KaggleApi

def download_credit_dataset():
    api = KaggleApi()
    api.authenticate()
    
    dataset_identifier = 'laotse/credit-risk-dataset'
    download_dir = 'data/raw'
    
    os.makedirs(download_dir, exist_ok=True)
    print(f'Downloading {dataset_identifier} from Kaggle...')
    api.dataset_download_files(dataset_identifier, path=download_dir, unzip=False)
    
    zip_path = os.path.join(download_dir, 'credit-risk-dataset.zip')
    if os.path.exists(zip_path):
        print('Extracting dataset archive content...')
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(download_dir)
        os.remove(zip_path)
        print(f'Data ready. Extracted files located at: {download_dir}/')
    else:
        print('Error: Target archive download failed.')

if __name__ == '__main__':
    download_credit_dataset()