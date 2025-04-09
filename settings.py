import os


MODEL_DIR = "models"
LORA_DIR = "lora"


def get_model(model_name):
    return os.path.join(MODEL_DIR, model_name)

def get_lora(model_name):
    return os.path.join(LORA_DIR, model_name)