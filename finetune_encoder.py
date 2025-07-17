#%%
# Für sicheres Parsing von JSON-Strings als Python-Literale (Fallback-Parser)
import ast

# Für JSON-Datei-Operationen (Laden der synthetischen Trainingsdaten)
import json

# Für DataFrame-Operationen und Training-Log-Analyse
import pandas as pd

# Für Umgebungsvariablen-Zugriff (Hugging Face Token)
import os

# Für das Unterdrücken von SyntaxWarnings beim JSON-Parsing
import warnings

# Für das Zählen der Label-Verteilung (low/medium/high risk)
from collections import Counter

# Für die Erstellung von Hugging Face Datasets aus Listen
from datasets import Dataset

# Für das Laden von .env-Dateien mit API-Tokens
from dotenv import load_dotenv

# Für moderne Dateipfad-Operationen (Glob-Pattern für JSON-Files)
from pathlib import Path

# Hugging Face Transformers: Model, Training-Config, Trainer und Early Stopping
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer, EarlyStoppingCallback

# Für Visualisierung der Training/Validation Loss Kurven
import matplotlib.pyplot as plt

# Für die Tokenisierung der Command-Texte (DeBERTa Tokenizer)
from transformers import AutoTokenizer

# Für dynamisches Padding der Batches während des Trainings
from transformers import DataCollatorWithPadding


load_dotenv()
hf_token = os.getenv("HF_TOKEN")
#%% md
# # Laden und aufbereiten der Trainingsdaten
#%% md
# Erstellen der Labeldictionaries
#%%
id2label = {0: "low", 1: "medium", 2: "high"}
label2id = {"low": 0, "medium": 1, "high": 2}
#%% md
# trainingsdaten formatieren und codieren
#%%
data_list = []
label_counts = Counter()
parse_errors = 0
invalid_types = 0

p = Path("/home/m/PycharmProjects/thesis_playground/data/json_results")
for json_file in p.glob('*.json'):
    with json_file.open('r', encoding='utf-8') as f:
        try:
            file_data = json.load(f)
        except json.JSONDecodeError as e:
            print(f"❌ Skipping file {json_file.name} due to top-level JSON error: {e}")
            continue

    for id, json_data in file_data.items():
        if isinstance(json_data, str):
            try:
                json_data = json.loads(json_data)
            except json.JSONDecodeError:
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", SyntaxWarning)
                        json_data = ast.literal_eval(json_data)
                except Exception:
                    parse_errors += 1
                    continue

        if not isinstance(json_data, dict):
            invalid_types += 1
            continue

        label = label2id.get(json_data.get("risk_level"))
        text = json_data.get("command")

        if label is not None and text is not None:
            data_list.append({
                "label": label,
                "text": text
            })
            label_counts[label] += 1

# Summary
print(f"✅ Parsed entries: {len(data_list)}")
print(f"❌ Parse errors: {parse_errors}")
print(f"⚠️ Skipped non-dict entries: {invalid_types}")
print("\n📊 Label distribution:")
for label, count in label_counts.items():
    print(f"  {label}: {count}")

#%%
data_list
#%% md
# Konvertiere in Huggingface Dataset
#%%
dataset = Dataset.from_list(data_list)
#%% md
# # Preprocessing
#%%

tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")
#%%
def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True)

tokenized_datasets = dataset.map(preprocess_function, batched=True)
#%%
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
#%%
# Split: 99% train, 1% validation (or adjust as needed)
split_dataset = tokenized_datasets.train_test_split(test_size=0.01, seed=42)

train_dataset = split_dataset["train"]
eval_dataset = split_dataset["test"]

print(split_dataset["train"][0])
#%% md
# # Training
#%%
model = AutoModelForSequenceClassification.from_pretrained(
    "microsoft/deberta-v3-base", num_labels=3, id2label=id2label, label2id=label2id
)
#%%
trainer = Trainer(
    model=model,
    processing_class=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    args = TrainingArguments(
        output_dir="outputs",
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        gradient_accumulation_steps=2,
        warmup_steps=50,
        max_steps=1000,
        learning_rate=2e-4,
        logging_steps=1,
        eval_strategy="steps",       # Updated parameter
        eval_steps=10,               # Evaluate every 30 steps
        save_strategy="steps",
        save_steps=10,
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        seed=3407,
    )
)
#%%
trainer.train()
#%%
# Access the log history
log_history = trainer.state.log_history

# Convert to a DataFrame for easier handling
logs_df = pd.DataFrame(log_history)

# Filter out rows that contain training loss
train_loss = logs_df[logs_df["loss"].notnull()][["step", "loss"]]

# Filter out rows that contain evaluation loss
eval_loss = logs_df[logs_df["eval_loss"].notnull()][["step", "eval_loss"]]

# Plotting the losses
plt.figure(figsize=(10, 6))
plt.plot(train_loss["step"], train_loss["loss"], label="Training Loss")
plt.plot(eval_loss["step"], eval_loss["eval_loss"], label="Validation Loss")
plt.xlabel("Training Steps")
plt.ylabel("Loss")
plt.title("Training and Validation Loss over Time")
plt.legend()
plt.grid(True)
plt.show()
#%%
model.push_to_hub("terminAl-thesis-2025/deberta-v3-base-terminAl-guard", private=True, token=hf_token)
tokenizer.push_to_hub("terminAl-thesis-2025/deberta-v3-base-terminAl-guard", private=True, token=hf_token)
