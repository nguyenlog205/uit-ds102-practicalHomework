'''
This module is to extract sematic vectors of the 
students' responses (after being segmented).
'''
from utils.logger import get_logger
from word_segmentation import load_model, tokenize_stringInDatasets

from transformers import AutoModel, AutoTokenizer
import torch
import pandas as pd
import numpy as np
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

def extract_semanticVector(sentence, model: AutoModel, tokenizer: AutoTokenizer):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    input_ids = tokenizer.encode(sentence, return_tensors='pt').to(device)
    with torch.no_grad():
        outputs = model(input_ids)

    last_hidden_states = outputs[0]  # shape: (1, seq_len, 768)
    sentence_embedding = torch.mean(last_hidden_states, dim=1)  # shape: (1, 768)

    return sentence_embedding.squeeze().cpu()

def compute_semanticVectorForDataset(datasets: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame], 
                                     model: AutoModel, 
                                     tokenizer: AutoTokenizer) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame] | None:
    logger.info('Start computing semantic vectors for datasets ...')
    
    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    try:
        for i, dataset in enumerate(datasets, start=1):  # i starts from 1
            logger.info(f'{i-1}/3 datasets have been processed successfully!')
            
            dataset['semantic_vector'] = None

            for idx, row in dataset.iterrows():
                sentence = row['tokenized_sentence']
                embedding = extract_semanticVector(sentence, model, tokenizer)
                dataset.at[idx, 'semantic_vector'] = embedding.cpu().tolist()
                
    except Exception as e:
        logger.error(f'Error computing semantic vectors for datasets: {e}')
        return None
    
    return datasets

def save_processed_datasets(datasets: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame], output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    datasets[0].to_json(os.path.join(output_dir, 'train_semantic.json'), orient='records', force_ascii=False)
    datasets[1].to_json(os.path.join(output_dir, 'dev_semantic.json'), orient='records', force_ascii=False)
    datasets[2].to_json(os.path.join(output_dir, 'test_semantic.json'), orient='records', force_ascii=False)


def main():
    datasets = load_segmentedDataset()
    if datasets is None:
        return

    phobert, phobertTokenizer = load_phoBertModel()
    if phobert is None or phobertTokenizer is None:
        return
    
    result = compute_semanticVectorForDataset(datasets=datasets, model=phobert, tokenizer=phobertTokenizer)

    if result:
        save_processed_datasets(result, output_dir='lesson-05/data/semanticVectors')
        logger.success('Semantic vector computation and saving completed successfully!')



if __name__ == "__main__":
    main()