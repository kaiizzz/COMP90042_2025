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

display_sample()

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

# display_sample()

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

# text tokenization
def tokenize_text(text):
    return text.split(" ")

train_df['claim_text'] = train_df['claim_text'].apply(tokenize_text)
dev_df['claim_text'] = dev_df['claim_text'].apply(tokenize_text)
dev_baseline_df['claim_text'] = dev_baseline_df['claim_text'].apply(tokenize_text)
test_df['claim_text'] = test_df['claim_text'].apply(tokenize_text)

evidence_df['value'] = evidence_df['value'].apply(tokenize_text)

display_sample()

## save into csv files
train_df.to_csv('data/train.csv', index=False)
dev_df.to_csv('data/dev.csv', index=False)
dev_baseline_df.to_csv('data/dev_baseline.csv', index=False)
test_df.to_csv('data/test.csv', index=False)
evidence_df.to_csv('data/evidence.csv', index=False)

################################# Doc2Vec ##################################
# Create TaggedDocument objects
train_tagged = [TaggedDocument(words=row['claim_text'], tags=[str(i)]) for i, row in train_df.iterrows()]
dev_tagged = [TaggedDocument(words=row['claim_text'], tags=[str(i)]) for i, row in dev_df.iterrows()]
dev_baseline_tagged = [TaggedDocument(words=row['claim_text'], tags=[str(i)]) for i, row in dev_baseline_df.iterrows()]
test_tagged = [TaggedDocument(words=row['claim_text'], tags=[str(i)]) for i, row in test_df.iterrows()]
evidence_tagged = [TaggedDocument(words=row['value'], tags=[row['key']]) for _, row in evidence_df.iterrows()]

# Create Doc2Vec model
model = Doc2Vec(vector_size=100, window=5, min_count=2, workers=4, epochs=40)
all_tagged = train_tagged + dev_tagged + dev_baseline_tagged + evidence_tagged

# Build vocabulary
model.build_vocab(all_tagged)

# Train the model
model.train(all_tagged, total_examples=model.corpus_count, epochs=model.epochs)

# GET VECTORS
def get_trained_vectors(model, tagged_data):
    vectors = []
    for doc in tagged_data:
        vectors.append(model.dv[doc.tags[0]])
    return vectors

model.save("doc2vec.model")
test_vectors = [model.infer_vector(doc.words) for doc in test_tagged]
train_vectors = get_trained_vectors(model, train_tagged)
np.save('train_vectors.npy', train_vectors)
np.save('test_vectors.npy', test_vectors)


