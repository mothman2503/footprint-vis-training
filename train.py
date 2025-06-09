
from datasets import Dataset
from transformers import BertTokenizerFast, BertForSequenceClassification, TrainingArguments, Trainer
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import evaluate
import os

# Load and prepare dataset
df = pd.read_csv("final_balanced_multilingual_dataset.csv")
df = df.rename(columns={"query_or_title": "text", "iab_label": "label"})

# Encode labels
unique_labels = sorted(df['label'].unique())
label2id = {label: i for i, label in enumerate(unique_labels)}
id2label = {i: label for label, i in label2id.items()}
df['label'] = df['label'].map(label2id)

# Split into train/val
train_df, val_df = train_test_split(df, test_size=0.1, stratify=df['label'], random_state=42)
train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(val_df)

# Tokenizer
tokenizer = BertTokenizerFast.from_pretrained("bert-base-multilingual-cased")
def preprocess(example):
    return tokenizer(example["text"], padding="max_length", truncation=True, max_length=128)
train_dataset = train_dataset.map(preprocess, batched=True)
val_dataset = val_dataset.map(preprocess, batched=True)

# Model
model = BertForSequenceClassification.from_pretrained(
    "bert-base-multilingual-cased",
    num_labels=len(label2id),
    id2label=id2label,
    label2id=label2id
)

# Metrics
accuracy = evaluate.load("accuracy")
f1 = evaluate.load("f1")
def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=1)
    return {
        "accuracy": accuracy.compute(predictions=preds, references=p.label_ids)["accuracy"],
        "f1": f1.compute(predictions=preds, references=p.label_ids, average="weighted")["f1"],
    }

# Training args
training_args = TrainingArguments(
    output_dir="./output",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    num_train_epochs=3,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

# Train and save
trainer.train()
trainer.save_model("./output/final_model")
tokenizer.save_pretrained("./output/final_model")
