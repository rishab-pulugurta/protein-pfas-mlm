# train.py
import os
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling
import wandb
from config import *
from dataset import PolymerProteinDataset
from model import PolymerProteinModel

# Initialize Weights and Biases
wandb.init(project=WANDB_PROJECT, entity=WANDB_ENTITY)

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = PolymerProteinModel(MODEL_NAME, UNFREEZE_LAYERS, USE_LORA)

# Add [SEP] token if not present
if '[SEP]' not in tokenizer.vocab:
    tokenizer.add_special_tokens({'sep_token': '[SEP]'})
    model.base_model.resize_token_embeddings(len(tokenizer))

# Load datasets
train_dataset = PolymerProteinDataset(TRAIN_CSV, tokenizer)
val_dataset = PolymerProteinDataset(VAL_CSV, tokenizer)

# Define data collator
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=1.0)

# Define training arguments
training_args = TrainingArguments(
    output_dir=CHECKPOINT_DIR,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    warmup_steps=WARMUP_STEPS,
    weight_decay=WEIGHT_DECAY,
    learning_rate=LEARNING_RATE,
    logging_dir='./logs',
    logging_steps=10,
    evaluation_strategy="steps",
    eval_steps=EVAL_STEPS,
    save_steps=SAVE_STEPS,
    load_best_model_at_end=True,
    report_to="wandb",
)

# Create Trainer instance
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=val_dataset
)

# Train the model
trainer.train()

# Save the model
trainer.save_model(CHECKPOINT_DIR)
tokenizer.save_pretrained(CHECKPOINT_DIR)