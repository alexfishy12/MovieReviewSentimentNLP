# MovieReviewSentimentNLP

<details>
<summary>Table of Contents</summary>
  
1. [Summary](#summary)
2. [Documentation](#documentation)
3. [Features](#features)
4. [Visuals](#visuals)
5. [Technologies](#technologies)
6. [What We Learned](#what-we-learned)
7. [Setup and Installation](#setup-and-installation)
8. [Usage](#usage)
9. [Code Examples](#code-examples)
10. [Contact](#contact)
11. [Acknowledgments](#acknowledgments)

</details>

## Summary
A simple webpage that is used to interface with a natural language processing machine learning model to predict sentiment analysis for the user-input movie review. 

## Documentation
- [Final Paper](web/_documents/Classifying_Movie_Reviews_with_Deep_Learning_Neural_Networks.pdf)
- [Final Presentation](web/_documents/CPS5801_07.pdf)

## Features
- **Natural Language Processing**: Uses a fine-tuned version of Google's BERT NLP model for sentiment analysis prediction.
- **Web UI**: An in-browser interface to make predictions using the machine learning model.

## Visuals
<p float="left">
  <img src="https://raw.githubusercontent.com/alexfishy12/MovieReviewSentimentNLP/main/web/_assets/webpagescreenshot.png" width="100%" />
</p>


## Technologies
- Front-end: HTML, CSS (with Bootstrap), Javascript
- Backend: Docker, Python, FastAPI, PyTorch ML Framework

## What We Learned
- **Fundamentals of PyTorch**: How to use PyTorch ML Framework to build data processing pipelines for ML training.
- **Containerization**: How to dockerize software including all of its dependencies for easier portability.
- **Implementing AI into a web application**: Used FastAPI to interface with Python backend to run model and produce sentiment analysis predictions.
- **Natural Language Processing**: How to extract, transform, and load natural language and process it to enable a machine learning model to perform sentiment analysis training and prediction

## Setup and Installation

### General Setup
1. Clone the repo: `git clone https://github.com/alexfishy12/MovieReviewSentimentNLP.git`
2. Download and install Docker:
   - Windows:  https://docs.docker.com/desktop/install/windows-install/
   - macOS: https://docs.docker.com/desktop/install/mac-install/
   - Linux: https://docs.docker.com/desktop/install/linux-install/
3. Create a `.env` file in the base project directory. The content of the file should look like the following (variable values are up to you, these values are what we used):

    ```env
    PORT_WEB=1140
    PORT_PYTHON=1141
    ```
4. If you plan on running directly on localhost, change the URL in the `script.js` calculate_sentiment function from `predict` to the following: `http://localhost:[PORT_PYTHON]/predict`.

### Model Training Setup

1. Download the IMDB dataset for movie reviews as a CSV from: https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews
    - Rename the downloaded file to `IMDB Dataset.csv` and place it in the same directory as the folder of this project. File structure should be as shown below:

        ```
        - IMDB Dataset.csv 
        - Base project directory folder
        ```

2. Create and activate virtual environment (in base project folder):
    - `pip install virtualenv`
    - `virtualenv [ENVIRONMENT_NAME]`
    - Windows: `.\[ENVIRONMENT_NAME]\Scripts\activate` || Unix/Mac: `source [ENVIRONMENT_NAME]/bin/activate`
4. Install dependencies into the virtual environment: `pip install python/requirements.txt`

6. To train and save the finetuned model to make predictions, you must run each cell in order in the file: [MovieReviewSentimentAnalysis.ipynb](python/MovieReviewSentimentAnalysis.ipynb). 
    - If successful, a new file named `new_finetuned_model.sav` will be created by the python notebook under the python folder. This is a file containing saved model weights for your finetuned model. At this point, the server backend will be ready to be built into a docker container.

## Usage
1. Run docker compose in the terminal of the base project directory: `docker compose up --build`
2. Visit `http://localhost:[PORT_WEB]` to interact with the running web application.

## Code Examples
- **Train and evaluate model over epochs**:

    ```python
    from tqdm.auto import tqdm
    from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
    import evaluate

    progress_bar = tqdm(range(num_training_steps))

    epoch_train_loss = []
    epoch_val_loss = []

    for epoch in range(num_epochs):
        # train
        train_loss = []
        model.train()
        for batch in train_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            train_loss.append(loss.item())
            loss.backward()

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)
        
        print("Epoch {} Avg train loss: {}".format(epoch, np.mean(train_loss)))
        epoch_train_loss.append(np.mean(train_loss))
        
        # eval
        val_loss = []
        model.eval()
        for batch in test_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = model(**batch)
            
            loss = outputs.loss
            val_loss.append(loss.item())
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)
        
        print("Epoch {} Avg val loss: {}".format(epoch, np.mean(val_loss)))
        epoch_val_loss.append(np.mean(val_loss))
    ```
- **Predict Sentiment (Positive or Negative)**: 

    ```python
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

    ```


## Contact
Alexander J. Fisher
  - **Email**: alexfisher0330@gmail.com
  - **LinkedIn**: https://www.linkedin.com/in/alexjfisher

Matthew Fernandez
  - **Email**: fermatth@kean.edu
  - **LinkedIn**: https://www.linkedin.com/in/mattfdez

Nicholas Moffa
  - **Email**: moffan@kean.edu
  - **LinkedIn**: https://www.linkedin.com/in/nicholas-moffa


## Acknowledgments
We would like to thank Kuan Huang, Ph.D. (khuang@kean.edu) who advised us during our graduate college course at Kean University (CPS 5801 - Advanced Artificial Intelligence) for this project.

---
