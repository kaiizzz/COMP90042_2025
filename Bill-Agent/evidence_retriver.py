### Author: Bill Zhu
### Date: 05/05/2025
### Description: This script is used to predict the claim labels and retrieve evidences for a given dataset using a pre-trained model.

import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load training set
with open('data/train-claims.json', 'r') as f:
    train_claims = json.load(f)

# Load development set
with open('data/dev-claims.json', 'r') as f:
    dev_claims = json.load(f)

# Load development baseline set
with open('data/dev-claims-baseline.json', 'r') as f:
    dev_baseline_claims = json.load(f)

# Load test set
with open('data/test-claims-unlabelled.json', 'r') as f:
    test_claims = json.load(f)

# Load evidence corpus
with open('data/evidence.json', 'r') as f:
    evidence = json.load(f)

print(f"Train claims: {len(train_claims)}")
print(f"Dev claims: {len(dev_claims)}")
print(f"Dev baseline claims: {len(dev_baseline_claims)}")
print(f"Test claims: {len(test_claims)}")
print(f"Evidence passages: {len(evidence)}")

### do TF-IDK vectorization for the evidence corpus
ev_ids = list(evidence.keys())
ev_texts = [evidence[ev_id]['text'] for ev_id in ev_ids]

vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
evidence_tfidf = vectorizer.fit_transform(ev_texts)

top_n = 6

for cid, claim_obj in dev_claims.items():
    claim_text = claim_obj['claim_text']
    
    # Transform the claim into TF-IDF vector
    claim_tfidf = vectorizer.transform([claim_text])
    
    # Compute cosine similarity between claim and all evidence
    similarities = cosine_similarity(claim_tfidf, evidence_tfidf).flatten()
    
    # Get top_k indices
    top_indices = np.argsort(similarities)[::-1][:top_n]
    
    # Get top evidence IDs
    top_evidence_ids = [ev_ids[idx] for idx in top_indices]
    
    # Add to dev_claims for later use
    dev_claims[cid]['predicted_evidences'] = top_evidence_ids

    
for cid, claim_obj in list(dev_claims.items())[:3]:
    print(f"Claim: {claim_obj['claim_text']}")
    print("Top evidence passages:")
    for eid in claim_obj['predicted_evidences']:
        print(f"- {evidence[eid]}")
    print()