from utils.logger import get_logger
from word_segmentation import load_model, tokenize_stringInDatasets

from transformers import AutoTokenizer, RobertaForMaskedLM
import torch