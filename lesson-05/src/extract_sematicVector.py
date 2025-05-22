from utils.logger import get_logger
from word_segmentation import load_model, tokenize_stringInDatasets

from transformers import AutoModel, AutoTokenizer
import torch


# Set logger
logger = get_logger('./lesson-05/log/compile_task02.log')

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
    # Load pre-works for upcoming task
    try:
        logger.subinfo('Start loading py_vncorenlp')
        model = load_model(r'C:\Users\VICTUS\Documents\developer\uit-practicalLesson-sML\lesson-05\model')
        if model is None:
            return None
        logger.success('Loading py_vncorenlp model has been completed successfully!')
    except Exception as e:
        logger.error(f'Error loading py_vncorenlp model: {e}')
        return
    
    # Load PhoBERT model
    phobert, phobertTokenizer = load_phoBertModel()
    


if __name__ == "__main__":
    main()