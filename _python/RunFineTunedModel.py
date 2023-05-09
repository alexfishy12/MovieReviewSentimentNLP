#!/usr/bin/python3

print("Content-Type: application/json\n\n")

# import dependencies
import joblib
import numpy as np
import pandas as pd
import torch
import cgi
import json
import seaborn as sns

# initialize json_response
json_response = {'response':{}, 'errors':[]}

# load fine-tuned model
model, tokenizer = joblib.load("./new_finetuned_model.sav")
model.to("cpu")

# get review from form input
form = cgi.FieldStorage()
review = None

if 'review_text' in form:
    review = form['review_text'].value
else:
    #review = str(input("Please enter a review..."))
    json_response['errors'].append("'review_text' not found in form data")
    print(json.dumps(json_response))
    quit()
    
    

label_names = ["negative", "positive"]

# tokenize and get a prediction
tokens = torch.tensor(tokenizer.encode(review, return_tensors='pt', add_special_tokens=True))
output = model(tokens, output_attentions=True)
attention = output.attentions
logits = output.logits
predictions = int(torch.argmax(logits))
probabilities = torch.softmax(logits, dim=1)

# return the probabilities for all classes
class_probabilities = {}
for i, label in enumerate(label_names):
    class_probabilities[label] = round(float(probabilities[0][i]), 4)

json_response['response']['probabilities'] = class_probabilities

# return a prediction for the received argument
json_response['response']['predictions'] = predictions

print(json.dumps(json_response))