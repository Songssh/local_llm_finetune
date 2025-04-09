from transformers import AutoTokenizer, Gemma3ForConditionalGeneration, AutoModelForCausalLM
from transformers import BitsAndBytesConfig

from peft import LoraConfig, get_peft_model, PeftModel, PeftConfig
from trl import SFTTrainer, SFTConfig

from datathing import load_dataset_from_json

import torch
import os

from settings import get_model, get_lora




def generate_train_data(example):
    prompt_list = []
    for i in range(len(example['instruction'])):
        prompt = r"""<bos><start_of_turn>user
dc 말투로 대댓글을 작성해 주세요.:
{}<end_of_turn>
<start_of_turn>model
{}<end_of_turn><eos>""".format(example['instruction'][i], example['output'][i])
        prompt_list.append(prompt)
    return {'train_data': prompt_list}



def finetune_model(model_path, lora_model, qlora = False, is_gemma = False):
    if qlora:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
        if is_gemma:
            model = Gemma3ForConditionalGeneration.from_pretrained(model_path, local_files_only=True, quantization_config=bnb_config)
        else:
            model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True, quantization_config=bnb_config)
    else:
        if is_gemma:
            model = Gemma3ForConditionalGeneration.from_pretrained(model_path, local_files_only=True)
        else:
            model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True)

    dataset = load_dataset_from_json("result.json")
    tokenized_datasets = dataset.map(generate_train_data, batched=True)
    #train_data = tokenized_datasets['train_data']

    #train_data = tokenized_datasets['train_data']

    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, add_bos=True)
    #tokenizer.padding_side = "right"

    def tokenize_train_data(example):
        return tokenizer(example['train_data'], padding="max_length", truncation=True, max_length=128)
    
    def tokenize_function(examples):
        inputs = tokenizer(examples["instruction"], padding="max_length", truncation=True, max_length=128)
        outputs = tokenizer(examples["output"], padding="max_length", truncation=True, max_length=128)
        inputs["labels"] = outputs["input_ids"]  # 모델이 "output"을 예측하도록 labels 추가
        return inputs
    
    train_data = tokenized_datasets.map(tokenize_function, batched=True)


    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias = 'none',
        task_type="CAUSAL_LM",
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
        modules_to_save=['embed_tokens', "lm_head"] # it works for gemma3, but not for other?
    )
    model = get_peft_model(model, lora_config)

    trainer = SFTTrainer(
        model = model,
        #tokenizer = tokenizer,
        train_dataset = train_data,
        eval_dataset = None, # Can set up evaluation!
        args = SFTConfig(
            dataset_text_field = "text",
            per_device_train_batch_size = 2,
            gradient_accumulation_steps = 4, # Use GA to mimic batch size!
            warmup_steps = 5,
            #num_train_epochs = 2, # Set this for 1 full training run.
            max_steps = 100,
            learning_rate = 2e-5, # Reduce to 2e-5 for long training runs 2e-4 for short ones.
            logging_steps = 1,
            optim = "adamw_8bit",
            weight_decay = 0.01,
            lr_scheduler_type = "linear",
            seed = 3407,
            report_to = "none", # Use this for WandB etc
        ),
    )
    trainer.train()
    trainer.model.save_pretrained(lora_model)
    #tokenizer.save_pretrained(out_model)
    return lora_model


def merge_lora_weights(model_file, lora, result_model, is_gemma = False):
    if is_gemma:
        base_model = Gemma3ForConditionalGeneration.from_pretrained(model_file, device_map='auto', torch_dtype=torch.float16)
    else:
        base_model = AutoModelForCausalLM.from_pretrained(model_file, device_map='auto', torch_dtype=torch.float16)
    # Load the LoRA weights
    model = PeftModel.from_pretrained(base_model, lora, device_map='auto', torch_dtype=torch.float16)

    # Merge LoRA weights into the base model
    model = model.merge_and_unload()

    tokenizer = AutoTokenizer.from_pretrained(model_file, local_files_only=True, add_bos=True)

    # Save the merged model
    model.save_pretrained(result_model)
    tokenizer.save_pretrained(result_model)
    return result_model



def main():
    model = get_model("gemma3_4b")
    is_gemma = True
    lora_model = get_lora('test3')
    result_model = get_model('gemma_dc_test3')


    #lora_model = finetune_model(model, lora_model, qlora=True, is_gemma=True)
    lora_model = finetune_model(model, lora_model, qlora=True, is_gemma=is_gemma)

    result_model = merge_lora_weights(model, lora_model, result_model, is_gemma=is_gemma)
    print("Model saved to:", result_model)


    
def test2():
    model = get_model("gemma3_4b_un")
    is_gemma = False
    lora_model = get_lora('test1')
    result_model = get_model('gemma_dc_test')
    result_model = merge_lora_weights(model, lora_model, result_model, is_gemma=is_gemma)
    print("Model saved to:", result_model)


def test():
    dataset = load_dataset_from_json("result.json")
    tokenized_datasets = dataset.map(generate_train_data, batched=True)
    print(tokenized_datasets[0]['train_data'])

if __name__ == "__main__":
    main()
    #test2()