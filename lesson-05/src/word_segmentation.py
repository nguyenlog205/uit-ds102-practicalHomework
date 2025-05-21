from typing import Optional
from utils.logger import get_logger
import py_vncorenlp
import pandas as pd
import json



# Setup logger 
logger = get_logger('./lesson-05/log/compile.log')

# Loading models

def load_model(model_path: str) -> Optional[py_vncorenlp.VnCoreNLP]:
    '''
    Downloads and loads the VnCoreNLP model.

    Args:
        model_path (str): Absolute path where the model should be downloaded and loaded from.
    
    Returns:
        An instance of py_vncorenlp.VnCoreNLP or None if failed.
    '''
    try:
        logger.info('Downloading and loading VnCoreNLP model...')
        py_vncorenlp.download_model(save_dir=model_path)
        model = py_vncorenlp.VnCoreNLP(save_dir=model_path, annotators=["wseg", "pos", "ner", "parse"])
        logger.info('VnCoreNLP model loaded successfully.')
        return model
    except Exception as e:
        logger.error(f'Error loading VnCoreNLP model: {e}')
        return None

model = load_model(vncorenlpPath='', outputPath='./lesson-05/model')


'''
    # Annotate a raw corpus
    # model.annotate_file(input_file="/absolute/path/to/input/file", output_file="/absolute/path/to/output/file")

    # Annotate a raw text
    model.print_out(
        model.annotate_text("Ông Nguyễn Khắc Chúc  đang làm việc tại Đại học Quốc gia Hà Nội. Bà Lan, vợ ông Chúc, cũng làm việc tại đây.")
    )
'''

























def load_dataset() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame] | None:
    '''
    This function loads dataset for further analysis. It returns train, dev, test respectively.
    '''

    logger.info('Start loading datasets ...')
    
    # train dataset 
    try:  
        df_train = pd.read_json('./lesson-05/data/UIT-VSFC-train.json', lines=False)
        print(df_train.head())
    except Exception as e:
        logger.error(f'Error loading train dataset: {e}')
        return None
    
    # dev dataset 
    try:
        df_dev = pd.read_json('./lesson-05/data/UIT-VSFC-dev.json', lines=False)
        print(df_dev.head())
    except Exception as e:
        logger.error(f'Error loading dev dataset: {e}')
        return None
    
    # test dataset 
    try:
        df_test = pd.read_json('./lesson-05/data/UIT-VSFC-test.json', lines=False)
        print(df_test.head())
    except Exception as e:
        logger.error(f'Error loading test dataset: {e}')
        return None
    
    logger.success('Loading datasets has been completed successfully!')

    return df_train, df_dev, df_test

def tokenize_string(string: str) -> list[str] | None:
    
    return None

def tokenize_stringInDatasets(datasets: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    '''
    This function is to return the tokenized datasets.
    Respectively train, dev, test datasets.
    '''
    tokenized_df_train = {}
    tokenized_df_dev   = {}
    tokenized_df_test  = {}



def main():
    datasets = load_dataset()
    if datasets is None:
        logger.error('The pipeline has been interrupted just now!')

if __name__ == '__main__':
    main()