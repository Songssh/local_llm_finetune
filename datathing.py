
from datasets import Dataset
import json

"""
[
    {
        "instruction": "애갤 인날먹단 단장 기상",
        "input": "",
        "output": "애갤 인날먹단 단장 기상"
    },
    {
        "instruction": "이 새끼 뭔데 평점도 줌",
        "input": "",
        "output": "꽝스킬 애니 리뷰 한번 해보라니까 점수까지 쳐주네"
    },
    {
        "instruction": "저도 가끔 gpt랑 놂",
        "input": "",
        "output": "뭐하고 놂"
    },

]

"""



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
