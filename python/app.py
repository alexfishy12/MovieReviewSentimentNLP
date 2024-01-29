# import dependencies
import joblib
import torch
import cgi
from fastapi import FastAPI, Form, File, UploadFile
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from typing_extensions import Annotated
import uvicorn

app = FastAPI()

origins = [
    "http://localhost:*",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# initialize json_response
json_response = {'response':{}, 'errors':[]}

# load fine-tuned model
model, tokenizer = joblib.load("new_finetuned_model.sav")
model.to("cpu")

@app.post("/predict")
async def predict(review_text: str = Form(...)):
    try:
        # get review from form input
        review = None

        if review_text:
            review = review_text
        else:
            #review = str(input("Please enter a review..."))
            json_response['errors'].append("'review_text' not found in form data")
            return json_response
            
        label_names = ["negative", "positive"]

        # tokenize and get a prediction
        tokens = torch.tensor(tokenizer.encode(review, return_tensors='pt', add_special_tokens=True))
        output = model(tokens, output_attentions=True)
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
    except Exception as e:
        return {'status': 'error', 'data': 'There was an error uploading the file: ' + str(e)}
    finally:
        return json_response

if __name__ == '__main__':
    uvicorn.run(app, port=3000, host="0.0.0.0")
