from utils.logger import get_logger
import py_vncorenlp
import pandas as pd
import json
import os

# Define BASE_DIR at the top for reusability
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Setup logger 
logger = get_logger('./lesson-05/log/compile.log')

# Loading models ========================================================================================================
def load_model(modelPath: str) -> py_vncorenlp.VnCoreNLP | None:
    '''
    Downloads (if needed) and loads the VnCoreNLP model.

    Args:
        modelPath (str): Absolute path where the model should be downloaded and loaded from.
    
    Returns:
        An instance of py_vncorenlp.VnCoreNLP or None if failed.
    '''

    try:
        logger.info('Start downloading py_vncorenlp model ...')
        py_vncorenlp.download_model(annotators=["wseg"], save_dir=modelPath)
    except Exception as e:
        logger.warning(f'Error downloading py_vncorenlp model: {e}')
    
    try:
        # model = py_vncorenlp.VnCoreNLP(annotators=["wseg", "pos", "ner", "parse"], save_dir=f'{modelPath}/models/VnCoreNLP')
        model = py_vncorenlp.VnCoreNLP(save_dir=modelPath)
    except Exception as e:
            logger.error(f'Error loading py_vncorenlp model: {e}')
            return None
    
    return model

# Load dataset ===================================================================================================================
def load_dataset() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame] | None:
    '''
    This function loads dataset for further analysis. It returns train, dev, test respectively.
    '''

    logger.info('Start loading datasets ...')
    # Use BASE_DIR for dataset_dir
    dataset_dir = os.path.join(BASE_DIR, 'data', 'originalDatasets')
    
    # train dataset 
    try:  
        df_train = pd.read_json(os.path.join(dataset_dir, 'UIT_VSFC_train.json'))
        print(df_train.head())
    except Exception as e:
        logger.error(f'Error loading train dataset: {e}')
        return None
    
    # dev dataset 
    try:
        df_dev = pd.read_json(os.path.join(dataset_dir, 'UIT_VSFC_dev.json'))
        print(df_dev.head())
    except Exception as e:
        logger.error(f'Error loading dev dataset: {e}')
        return None
    
    # test dataset 
    try:
        df_test = pd.read_json(os.path.join(dataset_dir, 'UIT_VSFC_test.json'))
        print(df_test.head())
    except Exception as e:
        logger.error(f'Error loading test dataset: {e}')
        return None
    
    logger.success('Loading datasets has been completed successfully!')

    return df_train, df_dev, df_test

def tokenize_string(string: str, model: py_vncorenlp.VnCoreNLP) -> list[str] | None:
    '''
    This function tokenizes a given string using the provided VnCoreNLP model.
    '''
    if not isinstance(string, str) or not string.strip(): # Handle non-string or empty strings
        return []

    try:
        tokens = model.word_segment(string)
        if tokens and len(tokens) > 0:
            return tokens[0]
        return []
    except Exception as e:
        logger.error(f'Error tokenizing string "{string}": {e}')
        return None

def tokenize_stringInDatasets(datasets: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame], model: py_vncorenlp.VnCoreNLP) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame] | None:
    '''
    This function tokenizes the 'sentence' column in the provided datasets,
    creates a new 'tokenized_sentence' column, and then drops the original
    'sentence' column.
    Respectively returns tokenized train, dev, test datasets.
    '''
    df_train, df_dev, df_test = datasets
    dataframes = {'train': df_train, 'dev': df_dev, 'test': df_test}
    processed_dataframes = {}

    logger.info("Starting tokenization of datasets...")
    try:
        for name, df in dataframes.items():
            # Check if 'sentence' column exists in the current DataFrame
            if 'sentence' not in df.columns:
                logger.error(f"{name.capitalize()} DataFrame must contain a 'sentence' column for tokenization.")
                return None

            # Apply the tokenize_string function to the 'sentence' column
            # and handle potential NaN or non-string values gracefully
            df['tokenized_sentence'] = df['sentence'].apply(
                lambda x: tokenize_string(x, model) # tokenize_string now handles non-string/NaN
            )

            # Drop the original 'sentence' column after tokenization
            df = df.drop(columns=['sentence'])
            processed_dataframes[name] = df

        logger.success("Dataset tokenization completed successfully!")
    except Exception as e:
        logger.error(f"An unexpected error occurred during dataset tokenization: {e}")
        return None

    return processed_dataframes['train'], processed_dataframes['dev'], processed_dataframes['test']

def save_dataset(df_train_tokenized, df_dev_tokenized, df_test_tokenized):
    try:
        logger.info('Start saving word-segmented datasets ...')

        # Use BASE_DIR for the output path
        output_dir = os.path.join(BASE_DIR, 'data', 'wordSegmentedDatasets')

        # Create the directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Save JSON files with pretty formatting (indent=4) and orient='records'
        df_train_tokenized.to_json(os.path.join(output_dir, 'train.json'), force_ascii=False, indent=4, orient='records')
        df_dev_tokenized.to_json(os.path.join(output_dir, 'dev.json'), force_ascii=False, indent=4, orient='records')
        df_test_tokenized.to_json(os.path.join(output_dir, 'test.json'), force_ascii=False, indent=4, orient='records')

        logger.info('Datasets saved successfully.')

    except FileNotFoundError:
        logger.error('File not found error!')
    except Exception as e:
        logger.error(f'Error saving datasets: {e}')
        return None
    
    
# Main execution ======================================================================================================
def main():  
    logger.info("Starting main pipeline...")

    # Load model
    # Use BASE_DIR for modelPath
    modelPath = os.path.join(BASE_DIR, 'model')
    os.makedirs(modelPath, exist_ok=True) # Ensure model directory exists for download - using os.makedirs as pathlib is excluded
    model = load_model(modelPath=modelPath)
    if model is None:
        logger.error('The pipeline has been interrupted: Model loading failed!')
        return

    # Load dataset
    datasets = load_dataset()
    if datasets is None:
        logger.error('The pipeline has been interrupted: Dataset loading failed!')
        return

    # Now, tokenize the entire datasets
    tokenized_datasets = tokenize_stringInDatasets(datasets, model)
    if tokenized_datasets is None:
        logger.error('The pipeline has been interrupted: Dataset tokenization failed!')
        return

    df_train_tokenized, df_dev_tokenized, df_test_tokenized = tokenized_datasets

    print("Tokenized training data head:")
    print(f"\n{df_train_tokenized.head()}") # Use f-string with \n for better logging format
    print("Tokenized development data head:")
    print(f"\n{df_dev_tokenized.head()}")
    print("Tokenized test data head:")
    print(f"\n{df_test_tokenized.head()}")

    save_dataset(df_dev_tokenized=df_dev_tokenized, df_test_tokenized=df_test_tokenized, df_train_tokenized=df_train_tokenized)

    logger.success("Pipeline executed successfully!")

if __name__ == '__main__':
    main()