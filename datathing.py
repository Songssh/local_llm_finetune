
from datasets import Dataset
import json

def load_dataset_from_json(json_file):
    """
    Load a dataset from a JSON file.
    
    Args:
        json_file (str): Path to the JSON file.
        
    Returns:
        Dataset: A Hugging Face Dataset object.
    """
    # Read the JSON file
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Convert to Dataset
    dataset = Dataset.from_list(data)

    return dataset