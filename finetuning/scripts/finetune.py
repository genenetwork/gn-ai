"""Script to finetune local model for GeneNetwork"""

import os

import pandas as pd
import torch
from datasets import Dataset, load_dataset
from dotenv import load_dotenv
from peft import (
    AutoPeftModelForCausalLM,
    LoraConfig,
    PeftModel,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedTokenizer,
    TrainingArguments,
)
from trl import SFTTrainer, SFTConfig


def format_example(
    example: dict[str, str],
    question_field: str,
    answer_field: str,
    tokenizer: PreTrainedTokenizer,
) -> dict[str, str]:
    question = example[question_field]
    answer = example[answer_field]
    prompt = [
        {"content": question, "role": "user"},
        {"content": answer, "role": "assistant"},
    ]
    formatted_prompt = tokenizer.apply_chat_template(prompt, tokenize=False)
    return {"data": formatted_prompt}


def prepare_data(
    dataset_path: str,
    question_field: str,
    answer_field: str,
    tokenizer: PreTrainedTokenizer,
    local_path: bool = False,
) -> Dataset:
    if local_path is False:
        dataset = load_dataset(dataset_path)
    else:
        df = pd.read_csv(dataset_path)
        dataset = Dataset.from_pandas(df)
    return dataset.map(
        format_example,
        fn_kwargs={
            "question_field": question_field,
            "answer_field": answer_field,
            "tokenizer": tokenizer,
        },
    )


def prepare_model(model_name: str) -> tuple[PeftModel, PreTrainedTokenizer]:
    peft_config = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.1, bias="none")
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        quantization_config=quant_config,
    )
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, peft_config)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer


def finetune(
    model_name: str,
    dataset_path: str,
    output_path: str,
    question_field: str,
    answer_field: str,
):
    model, tokenizer = prepare_model(model_name)
    dataset = prepare_data(dataset_path, question_field, answer_field, tokenizer)
    sft_config = SFTConfig(
    output_dir=output_path,
    dataset_text_field="data",
    bf16=False,
    fp16=False,
    )
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset["train"],
        args=sft_config,
        processing_class=tokenizer,
    )
    trainer.train()
    basename = model_name.split("/")[-1]
    finetuned_name = f"{output_path}/{basename}-lora-finetuned"
    trainer.model.save_pretrained(finetuned_name)
    new_model = AutoPeftModelForCausalLM(
        finetuned_name,
        device_map="auto",
    )
    merged_model = new_model.merge_and_unload()
    merged_model.save_pretrained(f"{output_path}/{basename}-lora-merged")


if __name__ == "__main__":
    load_dotenv()
    MODEL_NAME = os.environ["MODEL_NAME"]
    DATASET_PATH = os.environ["DATASET_PATH"]
    OUTPUT_PATH = os.environ["OUTPUT_PATH"]
    QUESTION_FIELD = os.environ["QUESTION_FIELD"]
    ANSWER_FIELD = os.environ["ANSWER_FIELD"]
    finetune(MODEL_NAME, DATASET_PATH, OUTPUT_PATH, QUESTION_FIELD, ANSWER_FIELD)
