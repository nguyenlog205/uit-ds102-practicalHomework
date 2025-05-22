from utils.logger import get_logger
import py_vncorenlp
import pandas as pd
import json
import os




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
    
    # train dataset 
    try:  
        df_train = pd.read_json('C:/Users/VICTUS/Documents/developer/uit-practicalLesson-sML/lesson-05/data/originalDataset/UIT-VSFC-train.json', lines=False)
        print(df_train.head())
    except Exception as e:
        logger.error(f'Error loading train dataset: {e}')
        return None
    
    # dev dataset 
    try:
        df_dev = pd.read_json('C:/Users/VICTUS/Documents/developer/uit-practicalLesson-sML/lesson-05/data/originalDataset/UIT-VSFC-dev.json', lines=False)
        print(df_dev.head())
    except Exception as e:
        logger.error(f'Error loading dev dataset: {e}')
        return None
    
    # test dataset 
    try:
        df_test = pd.read_json('C:/Users/VICTUS/Documents/developer/uit-practicalLesson-sML/lesson-05/data/originalDataset/UIT-VSFC-test.json', lines=False)
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
    This function tokenizes the 'sentence' column in the provided datasets.
    Respectively returns tokenized train, dev, test datasets.
    '''
    df_train, df_dev, df_test = datasets

    logger.info("Starting tokenization of datasets...")
    try:
        # Apply the tokenize_string function to the 'sentence' column
        # Handle cases where 'sentence' might be missing or empty for a row
        if 'sentence' not in df_train.columns:
            logger.error("Train DataFrame must contain a 'sentence' column for tokenization.")
            return None
        df_train['tokenized_sentence'] = df_train['sentence'].apply(lambda x: tokenize_string(x, model))

        if 'sentence' not in df_dev.columns:
            logger.error("Dev DataFrame must contain a 'sentence' column for tokenization.")
            return None
        df_dev['tokenized_sentence'] = df_dev['sentence'].apply(lambda x: tokenize_string(x, model))

        if 'sentence' not in df_test.columns:
            logger.error("Test DataFrame must contain a 'sentence' column for tokenization.")
            return None
        df_test['tokenized_sentence'] = df_test['sentence'].apply(lambda x: tokenize_string(x, model))

        logger.success("Dataset tokenization completed successfully!")
    except Exception as e:
        logger.error(f"An unexpected error occurred during dataset tokenization: {e}")
        return None

    return df_train, df_dev, df_test

def save_dataset(df_train_tokenized, df_dev_tokenized, df_test_tokenized):
    try:
        logger.info('Start saving word-segmented datasets ...')

        # Đường dẫn gốc tính từ file word_segmentation.py
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))  # lesson-05
        output_dir = os.path.join(base_dir, 'data', 'wordSegmentedDataset')

        # Tạo thư mục nếu chưa tồn tại
        os.makedirs(output_dir, exist_ok=True)

        # Lưu các file JSON
        df_train_tokenized.to_json(os.path.join(output_dir, 'train.json'), force_ascii=False)
        df_dev_tokenized.to_json(os.path.join(output_dir, 'dev.json'), force_ascii=False)
        df_test_tokenized.to_json(os.path.join(output_dir, 'test.json'), force_ascii=False)

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
    modelPath = 'C:/Users/VICTUS/Documents/developer/uit-practicalLesson-sML/lesson-05/model'
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

# python lesson-05/src/word_segmentation.py