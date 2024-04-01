#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@name: inference.py
@author: Finbarrs Oketunji
@contact: f@finbarrs.eu
@time: Monday January 15 02:00:00 2024
@desc: LLM Inference.
@run: python3 inference.py
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load the fine-tuned model and tokenizer
model_path = "gpt2_medium_fine_tuned_model"
model = AutoModelForCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Set the device to use for inference
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Function to generate text
def generate_text(prompt, max_length=100):
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    output = model.generate(input_ids, max_length=max_length, num_return_sequences=1)
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return generated_text

# Usage
prompt = "Once upon a time..."
generated_text = generate_text(prompt)
print("Generated text:", generated_text)