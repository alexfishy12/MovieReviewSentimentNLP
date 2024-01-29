# Use an official Python base image based on Linux
FROM python:3.10.11

# Set the working directory in the container
WORKDIR /movie-review-ai

# Copy the Python backend files to the container
COPY ./requirements.txt ./requirements.txt

# Create and activate the conda environment
# Assumes that environment.yml is part of the copied files
RUN pip install -r requirements.txt

COPY ./app.py ./app.py
COPY ./new_finetuned_model.sav ./new_finetuned_model.sav

# The code to run when container is started:
ENTRYPOINT ["python", "app.py"]