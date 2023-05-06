#!/usr/bin/python3
# coding: utf-8

# import dependencies
import joblib
import numpy as np
import pandas as pd
import torch
import sys

# load fine-tuned model
model, tokenizer = joblib.load("./finetuned_model.sav")
model.to("cpu")

# receive an argument to tokenize and get a prediction for
review = sys.argv[1]
tokens = tokenizer.encode(review, return_tensors='pt')
output = model(tokens)
logits = output.logits
predictions = int(torch.argmax(logits))

# return a prediction for the received argument
print(predictions)