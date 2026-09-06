"""Script to finetune local model for GeneNetwork"""

import json
import os
from pathlib import Path
from typing import Any

import optuna
import pandas as pd
import torch
from datasets import Dataset, concatenate_datasets, load_dataset
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
    pipeline,
)
from trl import SFTConfig, SFTTrainer


def format_example(
    example: dict[str, str],
    question_field: str,
    answer_field: str,
    tuning_instruction: str,
    tokenizer: PreTrainedTokenizer,
) -> dict[str, str]:
    question = example[question_field]
    answer = example[answer_field]
    prompt = [
        {
            "content": f"Your task is to: {tuning_instruction}\nThe question is: {question}",
            "role": "user",
        },
        {
            "content": f"I understand. The solution to the task is: {answer}",
            "role": "assistant",
        },
    ]
    formatted_prompt = tokenizer.apply_chat_template(prompt, tokenize=False)
    return {"data": formatted_prompt}


def prepare_data(
    dataset_path: str,
    question_field: str,
    answer_field: str,
    tuning_instruction: str,
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
            "tuning_instruction": tuning_instruction,
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


def prepare(
    model_name: str,
    dataset_path: str,
    question_field: str,
    answer_field: str,
    tuning_instruction: str,
    local_dataset: bool = False,
) -> tuple[Dataset, PreTrainedTokenizer, PeftModel]:
    model, tokenizer = prepare_model(model_name)
    dataset = prepare_data(
        dataset_path,
        question_field,
        answer_field,
        tuning_instruction,
        tokenizer,
        local_path=local_dataset,
    )
    return dataset, tokenizer, model


def search_objective(
    trial: int,
    instruction_set: Dataset,
    tokenizer: PreTrainedTokenizer,
    model: PeftModel,
    output_path: str,
) -> float:
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    batch_size = trial.suggest_categorical("batch_size", [2, 4, 8])
    epochs = trial.suggest_int("num_epochs", 1, 5)

    sft_config = SFTConfig(
        output_dir=f"{output_path}/search",
        dataset_text_field="data",
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        eval_strategy="steps",
        eval_steps=50,
        metric_for_best_model="eval_loss",
        bf16=False,
        fp16=False,
    )
    trainer = SFTTrainer(
        model=model,
        train_dataset=instruction_set["train"],
        eval_dataset=instruction_set["validation"],
        args=sft_config,
        processing_class=tokenizer,
    )
    train_result = trainer.train()
    return train_result.training_loss


def search(
    instruction_set: Dataset,
    tokenizer: PreTrainedTokenizer,
    model: PeftModel,
    output_path: str,
) -> dict[str, Any]:
    sampler = optuna.samplers.TPESampler(seed=100)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    wrapper_objective = lambda trial: search_objective(
        trial,
        instruction_set=instruction_set,
        tokenizer=tokenizer,
        model=model,
        output_path=output_path,
    )
    study.optimize(wrapper_objective, n_trials=10)
    return study.best_params


def finetune(
    instruction_set: Dataset,
    tokenizer: PreTrainedTokenizer,
    model: PeftModel,
    output_path: str,
    best_params: dict[str, Any],
) -> AutoModelForCausalLM:
    sft_config = SFTConfig(
        output_dir=f"{output_path}/best",
        dataset_text_field="data",
        learning_rate=best_params.get("learning_rate"),
        per_device_train_batch_size=best_params.get("batch_size"),
        num_train_epochs=best_params.get("epochs"),
        bf16=False,
        fp16=False,
    )
    trainer = SFTTrainer(
        model=model,
        train_dataset=concatenate_datasets(
            [instruction_set["train"], instruction_set["validation"]]
        ),
        args=sft_config,
        processing_class=tokenizer,
    )
    trainer.train()
    basename = model_name.split("/")[-1]
    finetuned_name = f"{output_path}/{basename}-lora-finetuned"
    trainer.model.save_pretrained(finetuned_name)
    new_model = AutoPeftModelForCausalLM.from_pretrained(
        finetuned_name,
        device_map="auto",
    )
    merged_model = new_model.merge_and_unload()
    merged_model.save_pretrained(f"{output_path}/{basename}-lora-merged")
    return merged_model


def test(
    instruction_set: Dataset,
    tokenizer: PreTrainedTokenizer,
    model: AutoModelForCausalLM,
) -> list[dict[str, str]]:
    pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer)
    test_results = []
    for instance in instruction_set["test"].select(range(100)):
        prompt = instance["data"]
        content = prompt.split(">")
        user_content = content[:3]
        assistant_content = content[3:]
        new_prompt = f"{'>'.join(user_content)}>"
        true_answer = ">".join(assistant_content)
        llm_answer = pipe(new_prompt)[0]["generated_text"]
        test_results.append(
            {
                "User prompt": new_prompt,
                "Model answer": llm_answer,
                "True answer": true_answer,
            }
        )
    return test_results


if __name__ == "__main__":
    load_dotenv()
    model_name = os.environ["MODEL_NAME"]
    dataset_path = os.environ["DATASET_PATH"]
    output_path = os.environ["OUTPUT_PATH"]
    question_field = os.environ["QUESTION_FIELD"]
    answer_field = os.environ["ANSWER_FIELD"]
    tuning_instruction = os.environ["TUNING_INSTRUCTION"]
    local_dataset = os.getenv("LOCAL_DATASET")

    torch.manual_seed(100)
    if local_dataset is None:
        dataset, tokenizer, model = prepare(
            model_name, dataset_path, question_field, answer_field, tuning_instruction
        )
    else:
        dataset, tokenizer, model = prepare(
            model_name,
            dataset_path,
            question_field,
            answer_field,
            tuning_instruction,
            local_dataset=True,
        )

    finetuned_path = f"{output_path}/{model_name.split('/')[-1]}-lora-merged"
    if not Path(finetuned_path).exists():
        best_params = search(dataset, tokenizer, model, output_path)
        finetuned_model = finetune(dataset, tokenizer, model, output_path, best_params)
    else:
        finetuned_model = AutoModelForCausalLM.from_pretrained(
            finetuned_path, device_map="auto"
        )

    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
    )
    base_results = test(dataset, tokenizer, base_model)
    finetuned_results = test(dataset, tokenizer, finetuned_model)

    with open(f"{output_path}/test_results.json", "w") as f:
        f.write(
            json.dumps(
                {
                    "Before finetuning": base_results,
                    "After finetuning": finetuned_results,
                },
                indent=4,
            )
        )
