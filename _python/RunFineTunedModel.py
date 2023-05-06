#!/usr/bin/python3

print("Content-Type: application/json\n\n")

# import dependencies
import joblib
import numpy as np
import pandas as pd
import torch
import cgi
import json

# initialize json_response
json_response = {'response':{}, 'errors':[]}

# load fine-tuned model
model, tokenizer = joblib.load("./finetuned_model.sav")
model.to("cpu")

# get review from form input
form = cgi.FieldStorage()
review = None

if 'review_text' in form:
    review = form['review_text'].value
else:
    json_response['errors'].append("'review_text' not found in form data")
    print(json.dumps(json_response))
    quit()


# tokenize and get a prediction
tokens = tokenizer.encode(review, return_tensors='pt')
output = model(tokens)
logits = output.logits
predictions = int(torch.argmax(logits))

# return a prediction for the received argument
json_response['response']['predictions'] = predictions

print(json.dumps(json_response))