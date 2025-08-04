
import os
import json
from datasets import Dataset, DatasetDict
import evaluate
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)

from accelerate import Accelerator
from peft import LoraConfig, get_peft_model, TaskType  # Import LoRA related modules
import wandb

def load_local_dataset(train_path, val_path):
    # Load train data
    with open(train_path, 'r', encoding='utf-8') as f:
        train_data = [json.loads(line) for line in f]
    
    # Load validation data
    with open(val_path, 'r', encoding='utf-8') as f:
        val_data = [json.loads(line) for line in f]
    
    # Create Dataset objects
    train_dataset = Dataset.from_list(train_data)
    val_dataset = Dataset.from_list(val_data)
    
    # Create DatasetDict
    return DatasetDict({
        "train": train_dataset,
        "validation": val_dataset
    })

def main():

    wandb.init(project="bia", name="qwen2.5-7b-lora-classification")

    # 1. Configuration
    model_name = "Qwen2.5-7B-Instruct"
    
    # 2. Load tokenizer & model
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    accelerator = Accelerator()
    
    # Load base model
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,
        torch_dtype="auto",
        device_map={'':accelerator.device},
        use_flash_attention_2=True
    )
    model.config.pad_token_id = tokenizer.eos_token_id

#    # 在模型加载后添加LoRA配置
#     lora_config = LoraConfig(
#         task_type=TaskType.SEQ_CLS,
#         r=8,  # LoRA rank
#         target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],  # 覆盖所有线性层
#         lora_alpha=32,  # 建议alpha是rank的2-4倍
#         lora_dropout=0.05,  # 较小的dropout防止过拟合
#         bias="none",  # 通常不训练bias
#     )

#     model = get_peft_model(model, lora_config)


    # Load datasets
    train_path = "merged_train.jsonl"
    val_path = "merged_test.jsonl"
    raw_datasets = load_local_dataset(train_path, val_path)
    

    # 4. Preprocessing
    def tokenize_single(ex):
        text = f"You need to determine whether the response adheres to this constraint. Output 1 if it does, otherwise output 0. Response: {ex['answer']} Constraint: {ex['question']}"
        tokenized = tokenizer(text, truncation=True, padding=True)
        tokenized["labels"] = ex["label"]
        return tokenized

    tokenized_datasets = raw_datasets.map(
        tokenize_single,
        remove_columns=raw_datasets["train"].column_names,
    )

    data_collator = DataCollatorWithPadding(tokenizer)

    # 5. Metrics
    import numpy as np

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = logits.argmax(axis=-1)
        accuracy = (preds == labels).mean()
        return {"accuracy": float(accuracy)}  # 需转为 Python 原生 float
    
    # 6. Training arguments
    training_args = TrainingArguments(
        output_dir="/nas/shared/kilab/hf-hub/cfbench",
        eval_strategy="epoch",
        save_strategy="epoch",
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        num_train_epochs=3,
        learning_rate=5e-6,
        logging_steps=10,
        fp16=True,
        gradient_accumulation_steps=1,
        save_total_limit=2,
        deepspeed="./deepspeed_config.json",
        optim="adamw_torch",         # Recommended for LoRA
        report_to="wandb",            # Disable logging if not needed
    )
    
    
    # 7. Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    
    # 8. Train & evaluate
    trainer.train()
    trainer.evaluate()
    


if __name__ == "__main__":
    main()
