# IMDB-Sentimental-Analysis
Multilayer Neural network for Sentimental Analysis of IMBD reviews using TF-IDF Vectorizer.

## Installation
Run ```pip3 install requirements.txt``` cmd to install the modules necessary for the code. 

## Data source
Download the data and place it in data directory
```https://ai.stanford.edu/~amaas/data/sentiment/```

## How to run the code
### Step1: Installation
Run the installation command
### Step2: Train the ANN and save the model
Run ```python3 train_NLP.py``` cmd to generate a new model and vectorizer from scratch.
The model as well as the tf-idf vectorizer model get saved in models directory.
### Step3: Test the models and calculate the accuracy
Run ```python3 test_NLP.py``` cmd to test the test data against the saved models.
Find the accuracy.

# Report
Read the report for more understanding of the solution.