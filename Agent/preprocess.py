### Author: Bill Zhu
### Date: 05/05/2025
### Description: This script is used to predict the claim labels and retrieve evidences for a given dataset using a pre-trained model.

import pandas as pd
import json
import re
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from gensim.models.doc2vec import TaggedDocument
from gensim.models import Doc2Vec
import numpy as np

stop_words = set(stopwords.words('english'))

# Load JSON files
with open('data/train-claims.json', 'r') as f:
    train_claims = json.load(f)

with open('data/dev-claims.json', 'r') as f:
    dev_claims = json.load(f)

with open('data/dev-claims-baseline.json', 'r') as f:
    dev_baseline_claims = json.load(f)

with open('data/test-claims-unlabelled.json', 'r') as f:
    test_claims = json.load(f)

with open('data/evidence.json', 'r') as f:
    evidence = json.load(f)

# Convert to DataFrames
train_df = pd.DataFrame(train_claims)
dev_df = pd.DataFrame(dev_claims)
dev_baseline_df = pd.DataFrame(dev_baseline_claims)
test_df = pd.DataFrame(test_claims)
evidence_df = pd.DataFrame(list(evidence.items()), columns=['key', 'value'])

train_df = pd.DataFrame(train_claims).transpose()
dev_df = pd.DataFrame(dev_claims).transpose()
dev_baseline_df = pd.DataFrame(dev_baseline_claims).transpose()
test_df = pd.DataFrame(test_claims).transpose()

# Display sample
def display_sample(n=5):
    print(train_df.head(n))
    print(dev_df.head(n))
    print(dev_baseline_df.head(n))
    print(test_df.head(n))
    print(evidence_df.head(n))

# display_sample() # ----------------------------------------------------------- Debugging
print("Loaded data")
print(f"train_df: {train_df.shape}")
print(f"dev_df: {dev_df.shape}")
print(f"dev_baseline_df: {dev_baseline_df.shape}")
print(f"test_df: {test_df.shape}")
print(f"evidence_df: {evidence_df.shape}")

print()
print("Start preprocessing...")
############################### PreProcessing ##################################
# text normalization
def normalize_text(text):
    # Convert to lowercase
    text = text.lower()
    # Keep only letters, numbers, and spaces
    text = re.sub(r'[^a-z0-9\s]', '', text)
    return text

train_df['claim_text'] = train_df['claim_text'].apply(normalize_text)
dev_df['claim_text'] = dev_df['claim_text'].apply(normalize_text)
dev_baseline_df['claim_text'] = dev_baseline_df['claim_text'].apply(normalize_text)
test_df['claim_text'] = test_df['claim_text'].apply(normalize_text)

evidence_df['value'] = evidence_df['value'].apply(normalize_text)

print(f"[STEP 1/4] Normalised text!")
# display_sample() # ----------------------------------------------------------- Debugging

# remove stop words
def remove_stopwords(text):
    words = text.split()
    filtered_words = [word for word in words if word.lower() not in stop_words]
    return ' '.join(filtered_words)

train_df['claim_text'] = train_df['claim_text'].apply(remove_stopwords)
dev_df['claim_text'] = dev_df['claim_text'].apply(remove_stopwords)
dev_baseline_df['claim_text'] = dev_baseline_df['claim_text'].apply(remove_stopwords)
test_df['claim_text'] = test_df['claim_text'].apply(remove_stopwords)

evidence_df['value'] = evidence_df['value'].apply(remove_stopwords)
print(f"[STEP 2/4] Removed stop words!")

# lemmatization
lemmatizer = WordNetLemmatizer()
def lemmatize_text(text):
    words = text.split()
    lemmatized_words = [lemmatizer.lemmatize(word) for word in words]
    return ' '.join(lemmatized_words)

train_df['claim_text'] = train_df['claim_text'].apply(lemmatize_text)
dev_df['claim_text'] = dev_df['claim_text'].apply(lemmatize_text)
dev_baseline_df['claim_text'] = dev_baseline_df['claim_text'].apply(lemmatize_text)
test_df['claim_text'] = test_df['claim_text'].apply(lemmatize_text)

evidence_df['value'] = evidence_df['value'].apply(lemmatize_text)

print(f"[STEP 3/4] Lemmatized text!")

# text tokenization
def tokenize_text(text):
    return text.split(" ")

train_df['claim_text'] = train_df['claim_text'].apply(tokenize_text)
dev_df['claim_text'] = dev_df['claim_text'].apply(tokenize_text)
dev_baseline_df['claim_text'] = dev_baseline_df['claim_text'].apply(tokenize_text)
test_df['claim_text'] = test_df['claim_text'].apply(tokenize_text)

evidence_df['value'] = evidence_df['value'].apply(tokenize_text)

print("[STEP 4/4] Tokenized text!")

# display_sample() # ----------------------------------------------------------- Debugging

train_df.to_json('processed/preprocessed_train.json', orient='index')
dev_df.to_json('processed/preprocessed_dev.json', orient='index')
dev_baseline_df.to_json('processed/preprocessed_dev_baseline.json', orient='index')
test_df.to_json('processed/preprocessed_test.json', orient='index')
evidence_df.to_json('processed/preprocessed_evidence.json', orient='records')
print(f"[FINISHED] Saved preprocessed data to JSON files...")


print("\nReady for vectorization!")

def return_df():
    return train_df, dev_df, dev_baseline_df, test_df, evidence_df

def return_original_df():
    return pd.DataFrame(train_claims), pd.DataFrame(dev_claims), pd.DataFrame(dev_baseline_claims), pd.DataFrame(test_claims), pd.DataFrame(list(evidence.items()), columns=['key', 'value'])