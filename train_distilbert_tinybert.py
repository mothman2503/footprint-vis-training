import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from transformers.trainer_utils import EvalPrediction
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score
from datasets import Dataset
import numpy as np
import argparse
import joblib

# --- CONFIG ---
MODEL_CHOICES = {
    "distilbert": "distilbert-base-uncased",
    "tinybert": "prajjwal1/bert-tiny",
    "bert": "bert-base-uncased"
}

# Fixed label list to ensure consistent label order
FIXED_LABEL_LIST = [
    "IAB1 Arts & Entertainment", "IAB2 Automotive", "IAB3 Business", "IAB4 Careers",
    "IAB5 Education", "IAB6 Family & Parenting", "IAB7 Health & Fitness", "IAB8 Food & Drink",
    "IAB9 Hobbies & Interests", "IAB10 Home & Garden", "IAB11 Law, Gov’t & Politics",
    "IAB12 News", "IAB13 Personal Finance", "IAB14 Society", "IAB15 Science", "IAB16 Pets",
    "IAB17 Sports", "IAB18 Style & Fashion", "IAB19 Technology & Computing", "IAB20 Travel",
    "IAB21 Real Estate", "IAB22 Shopping", "IAB23 Religion & Spirituality", "IAB24 Uncategorized"
]

# --- LOAD DATA ---
def load_and_prepare_data(train_path, val_path, test_path):
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)

    for df in [train_df, val_df, test_df]:
        df.drop(columns=["labels"], inplace=True, errors="ignore")

    le = LabelEncoder()
    le.fit(FIXED_LABEL_LIST)
    train_df["label"] = le.transform(train_df["iab_label"])
    val_df["label"] = le.transform(val_df["iab_label"])
    test_df["label"] = le.transform(test_df["iab_label"])

    return train_df, val_df, test_df, le

# --- TOKENIZATION ---
def tokenize_data(df, tokenizer):
    return tokenizer(df["text"].tolist(), truncation=True, padding=True, max_length=128)

# --- COMPUTE METRICS ---
def compute_metrics(eval_pred: EvalPrediction):
    preds = np.argmax(eval_pred.predictions, axis=1)
    labels = eval_pred.label_ids
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_macro": f1_score(labels, preds, average="macro")
    }

# --- MAIN TRAINING FUNCTION ---
def train_model(model_name_key, train_file, val_file, test_file, output_dir, use_weights):
    model_name = MODEL_CHOICES[model_name_key]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=len(FIXED_LABEL_LIST))

    train_df, val_df, test_df, label_encoder = load_and_prepare_data(train_file, val_file, test_file)

    joblib.dump(label_encoder, f"{output_dir}/label_encoder.joblib")

    train_dataset = Dataset.from_dict({**tokenize_data(train_df, tokenizer), "label": train_df["label"].tolist()})
    val_dataset = Dataset.from_dict({**tokenize_data(val_df, tokenizer), "label": val_df["label"].tolist()})
    test_dataset = Dataset.from_dict({**tokenize_data(test_df, tokenizer), "label": test_df["label"].tolist()})

    train_weights = torch.tensor(train_df["confidence"].values, dtype=torch.float32) if use_weights else torch.ones(len(train_df))

    def compute_loss_with_weights(model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        weights = inputs.pop("weights")
        outputs = model(**inputs)
        loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
        loss = loss_fct(outputs.logits, labels)
        weighted_loss = (loss * weights).mean()
        return (weighted_loss, outputs) if return_outputs else weighted_loss

    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        num_train_epochs=3,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        logging_dir=f"{output_dir}/logs",
        report_to="none"
    )

    class WeightedTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False):
            inputs["weights"] = train_weights[inputs["input_ids"].device.index if inputs["input_ids"].device.index else 0]
            return compute_loss_with_weights(model, inputs, return_outputs)

    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
    )

    trainer.train()

    metrics = trainer.evaluate(test_dataset)
    print("\nTest set evaluation:", metrics)

    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODEL_CHOICES.keys(), required=True)
    parser.add_argument("--train_file", default="train.csv")
    parser.add_argument("--val_file", default="val.csv")
    parser.add_argument("--test_file", default="test.csv")
    parser.add_argument("--output_dir", default="./model_output")
    parser.add_argument("--no_confidence_weights", action="store_true")
    args = parser.parse_args()

    use_weights = not args.no_confidence_weights
    train_model(args.model, args.train_file, args.val_file, args.test_file, args.output_dir, use_weights)