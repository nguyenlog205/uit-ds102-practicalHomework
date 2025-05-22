from utils.logger import get_logger
from word_segmentation import load_model, tokenize_stringInDatasets

from transformers import AutoModel, AutoTokenizer
import torch
import pandas as pd
import os


# Set logger
logger = get_logger()

# Load dataset and PhoBERT model
def load_segmentedDataset() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame] | None:
    '''
    This function loads segmented datasets from JSON files.

    Arg(s)
        None

    Output(s)
        Returns a tuple of three pandas DataFrames: (df_train, df_dev, df_test) if all files are found and loaded successfully.
        Returns None if any dataset file is missing or cannot be loaded.
    '''
    df_train = None
    df_dev = None
    df_test = None

    logger.info('Start loading segmented datasets ...')
    try:
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))  # lesson-05
        dataset_dir = os.path.join(base_dir, 'data', 'wordSegmentedDataset')

        df_train = pd.read_json(os.path.join(dataset_dir, 'train.json'))
        print(df_train.head())
        df_dev = pd.read_json(os.path.join(dataset_dir, 'dev.json'))
        print(df_dev.head())
        df_test = pd.read_json(os.path.join(dataset_dir, 'test.json'))
        print(df_test.head())
    except FileNotFoundError:
        logger.error('The datasets are not found!')
        return None
    except Exception as e:
        logger.error(f'An error occurred while loading datasets: {e}')
        return None
    
    logger.success('Loading datasets have been completed successfully!')
    return df_train, df_dev, df_test

def load_phoBertModel(typeOfModel: str = 'vinai/phobert-base-v2') -> tuple[AutoModel, AutoTokenizer] | None:
    '''
    This function is to load PhoBERT model. 
    
    Arg(s)
        `typeOfModel`: Refering to the type of model that is your expectation. (`vinai/phobert-base`, `vinai/phobert-base-v2`, and `vinai/phobert-large`)

    Output(s)
        Returns the chosen model and its tokenizer.
    '''
    logger.info(f'Start loading PhoBert model and ({typeOfModel})')
    try:
         phobert = AutoModel.from_pretrained(typeOfModel)
         tokenizer = AutoTokenizer.from_pretrained(typeOfModel)
    except Exception as e:
        logger.error(f'Error loading PhoBERT model: {e}')
        return None
    
    return phobert, tokenizer

def main():
    # Load segmented datasets for further works
    datasets = load_segmentedDataset()

    # Load PhoBERT model
    phobert, phobertTokenizer = load_phoBertModel()
    


if __name__ == "__main__":
    main()