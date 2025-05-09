import pandas as pd
import ast
import json
import numpy as np
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity

# === Load CSVs ===

csv_files = {
    'train': 'data/train.csv',
    'test': 'data/test.csv',
    'evidence': 'data/evidence.csv'
}

dfs = {}
for name, filepath in csv_files.items():
    df = pd.read_csv(filepath)

    if 'claim_text' in df.columns:
        df['claim_text'] = df['claim_text'].apply(lambda x: ast.literal_eval(x) if pd.notnull(x) else [])

    if 'evidences' in df.columns:
        df['evidences'] = df['evidences'].apply(lambda x: ast.literal_eval(x) if pd.notnull(x) else [])

    if name == 'evidence' and 'value' in df.columns:
        df['value'] = df['value'].apply(lambda x: ast.literal_eval(x) if pd.notnull(x) else [])

    dfs[name] = df
    print(f"Loaded {name}: {df.shape[0]} rows")

train_df = dfs['train']
test_df = dfs['test']
evidence_df = dfs['evidence']

# Load original test-claims-unlabelled.json
with open('data/test-claims-unlabelled.json', 'r') as f:
    original_test_claims = json.load(f)

# === Prepare TaggedDocuments ===

train_tagged = [TaggedDocument(words=row['claim_text'], tags=[str(i)]) for i, row in train_df.iterrows()]
test_tagged = [TaggedDocument(words=row['claim_text'], tags=[str(i)]) for i, row in test_df.iterrows()]

# === Load Doc2Vec model ===
model = Doc2Vec.load('doc2vec.model')

# === Infer vectors ===
train_vectors = [model.infer_vector(doc.words) for doc in train_tagged]
train_labels = train_df['claim_label'].tolist()
test_vectors = [model.infer_vector(doc.words) for doc in test_tagged]

# === Encode labels ===
label_encoder = LabelEncoder()
train_labels_encoded = label_encoder.fit_transform(train_labels)

# === Train classifier ===
clf = LogisticRegression(max_iter=100)
clf.fit(train_vectors, train_labels_encoded)

# === Predict on test set ===
predicted_labels_encoded = clf.predict(test_vectors)
predicted_labels = label_encoder.inverse_transform(predicted_labels_encoded)

# === Prepare evidence vectors and texts ===
evidence_vectors = {row['key']: model.infer_vector(row['value']) for _, row in evidence_df.iterrows()}
evidence_texts = {row['key']: set(row['value']) for _, row in evidence_df.iterrows()}

# === Prepare original test claim text ===
test_df_og = pd.DataFrame(original_test_claims).transpose()
test_claim_text = test_df_og['claim_text'].apply(lambda x: ' '.join(x) if isinstance(x, list) else x)

# === Save predictions with evidence ===
output_dict = {}

for i, row in test_df.iterrows():
    claim_id = f"claim-{i}"
    original_text = test_claim_text[i]
    predicted_label = predicted_labels[i]
    claim_words = set(row['claim_text'])

    # --- Filter evidence by keyword overlap ---
    filtered_evidence_keys = [
        key for key, words in evidence_texts.items()
        if claim_words & words  # non-empty intersection
    ]

    if not filtered_evidence_keys:
        # fallback: use all evidence if no keyword match
        filtered_evidence_keys = list(evidence_vectors.keys())

    # --- Compute similarity only on filtered set ---
    filtered_evidence_matrix = np.array([evidence_vectors[k] for k in filtered_evidence_keys])
    claim_vector = model.infer_vector(row['claim_text'])
    similarities = cosine_similarity([claim_vector], filtered_evidence_matrix)[0]

    # --- Get top 3 evidence IDs ---
    top_indices = similarities.argsort()[-3:][::-1]
    top_evidence_ids = [filtered_evidence_keys[idx] for idx in top_indices]

    output_dict[claim_id] = {
        "claim_text": original_text,
        "claim_label": predicted_label,
        "evidences": top_evidence_ids
    }

# === Save JSON ===
with open('doc2vec_test_predictions_with_evidence.json', 'w') as f:
    json.dump(output_dict, f, indent=2)

print("✅ Saved predictions with supporting evidences to doc2vec_test_predictions_with_evidence.json")
