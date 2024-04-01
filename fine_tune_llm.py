#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@name: fine_tune_llm.py
@author: Finbarrs Oketunji
@contact: f@finbarrs.eu
@time: Sunday January 14 23:22:00 2024
@desc: LLM Fine-tuning.
@run: python3 fine_tune_llm.py
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextDataset, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments

# Load the pre-trained model and tokenizer
model_name = "gpt2-medium"  # Replace with the desired pre-trained model
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load and preprocess the dataset
train_file = "dataset/train.txt"  # Replace with your training dataset file
train_dataset = TextDataset(
    tokenizer=tokenizer,
    file_path=train_file,
    block_size=128,  # Adjust the block size as needed
)

# Define the data collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,  # Set to True if using masked language modeling
)

# Define the training arguments
training_args = TrainingArguments(
    output_dir="output",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    save_steps=10_000,
    save_total_limit=2,
    prediction_loss_only=True,
)

# Create the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
)

# Fine-tune the model
trainer.train()

# Save the fine-tuned model
trainer.save_model("gpt2_medium_fine_tuned_model")