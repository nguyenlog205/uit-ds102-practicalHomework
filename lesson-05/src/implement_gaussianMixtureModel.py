import pandas as pd
import os


BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

def load_dataset() -> pd.DataFrame | None:
    """
    Load the semantic vector dataset from a JSON file.

    Returns:
        pd.DataFrame: If loaded successfully.
        None: If file not found or error occurs.
    """
    data_path = os.path.join(BASE_DIR, 'data', 'semanticVectors', 'dev_semantic.json')

    try:
        dataframe = pd.read_json(data_path)
        print('🟢 Loading dataset successfully!')
        return dataframe
    except FileNotFoundError:
        print('❌ Error: File not found at:', data_path)
    except Exception as e:
        print(f'❌ Error while loading dataset: {e}')
    
    return None

df = load_dataset()