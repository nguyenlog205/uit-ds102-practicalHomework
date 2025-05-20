from utils.logger import get_logger
import vncorenlp
import pandas as pd
import json


# Setup logger 
logger = get_logger('./lesson-05/log/compile.log')

def load_dataset() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame] | None:
    '''
    This function loads dataset for further analysis. It returns train, dev, test respectively.
    '''

    logger.info('Start loading dataset ...')
    
    # train dataset 
    try:  
        df_train = pd.read_json('./lesson-05/data/UIT-VSFC-train.json', lines=False)
        print(df_train.head())
    except Exception as e:
        logger.error(f'Error loading train dataset: {e}')
        return None
    
    try:
        df_dev = pd.read_json('./lesson-05/data/UIT-VSFC-dev.json', lines=False)
        print(df_dev.head())
    except Exception as e:
        logger.error(f'Error loading dev dataset: {e}')
        return None
    

    try:
        df_test = pd.read_json('./lesson-05/data/UIT-VSFC-test.json', lines=False)
        print(df_test.head())
    except Exception as e:
        logger.error(f'Error loading test dataset: {e}')
        return None
    
    logger.success('Loading dataset has been completed successfully!')

    return df_train, df_dev, df_test


def main():
    datasets = load_dataset()
    if datasets is not None:
        df_train, df_dev, df_test = datasets

if __name__ == '__main__':
    main()