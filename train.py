# 1. Environment Setup
try:
    import datasets
    import transformers
    import rouge_score
    import py7zr
    import accelerate
    import evaluate
except ImportError:
    print("Installing libraries...")
    import os
    os.system("pip install transformers datasets rouge_score py7zr accelerate evaluate nltk pandas matplotlib")

import torch
import nltk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import evaluate
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer

nltk.download('punkt')
nltk.download('punkt_tab')

# 2. Dataset Collection
print("Loading dataset...")
dataset = load_dataset("cnn_dailymail", "3.0.0")

# Optimization: Use a smaller subset for faster training (Lite Version)
print("Slicing dataset for faster training...")
dataset["train"] = dataset["train"].select(range(2000))
dataset["validation"] = dataset["validation"].select(range(500))

# 3. Data Preprocessing
model_checkpoint = "t5-small"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

max_input_length = 1024
max_target_length = 128
prefix = "summarize: "

def preprocess_function(examples):
    inputs = [prefix + doc for doc in examples["article"]]
    model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True)

    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples["highlights"], max_length=max_target_length, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

print("Tokenizing dataset (this might take a moment)...")
tokenized_datasets = dataset.map(preprocess_function, batched=True)

# 4. Model Setup
model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
batch_size = 16
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

# 5. Model Training
args = Seq2SeqTrainingArguments(
    f"{model_checkpoint}-finetuned-cnn",
    eval_strategy = "epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=3,
    predict_with_generate=True,
    fp16=True, # Improved speed on GPU
)

metric = evaluate.load("rouge")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
    decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]
    
    result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    # Extract results (evaluate returns floats directly)
    result = {key: value * 100 for key, value in result.items()}
    return {k: round(v, 4) for k, v in result.items()}

trainer = Seq2SeqTrainer(
    model,
    args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

print("Starting training...")
trainer.train()

# 6. Evaluation & Save
print("Evaluating model...")
trainer.evaluate()

print("Saving model...")
trainer.save_model("./my_summarization_model")
print("Done! You can now download the 'my_summarization_model' folder.")
