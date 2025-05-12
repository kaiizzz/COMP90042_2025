# # import pandas as pd
# # import ast
# # import json
# # import numpy as np
# # from gensim.models.doc2vec import Doc2Vec, TaggedDocument
# # from sklearn.linear_model import LogisticRegression
# # from sklearn.preprocessing import LabelEncoder
# # from sklearn.metrics.pairwise import cosine_similarity

# # # === Load CSVs ===

# # csv_files = {
# #     'train': 'data/train.csv',
# #     'test': 'data/test.csv',
# #     'evidence': 'data/evidence.csv'
# # }

# # dfs = {}
# # for name, filepath in csv_files.items():
# #     df = pd.read_csv(filepath)

# #     if 'claim_text' in df.columns:
# #         df['claim_text'] = df['claim_text'].apply(lambda x: ast.literal_eval(x) if pd.notnull(x) else [])

# #     if 'evidences' in df.columns:
# #         df['evidences'] = df['evidences'].apply(lambda x: ast.literal_eval(x) if pd.notnull(x) else [])

# #     if name == 'evidence' and 'value' in df.columns:
# #         df['value'] = df['value'].apply(lambda x: ast.literal_eval(x) if pd.notnull(x) else [])

# #     dfs[name] = df
# #     print(f"Loaded {name}: {df.shape[0]} rows")

# # train_df = dfs['train']
# # test_df = dfs['test']
# # evidence_df = dfs['evidence']

# # # Load original test-claims-unlabelled.json
# # with open('data/test-claims-unlabelled.json', 'r') as f:
# #     original_test_claims = json.load(f)

# # # === Prepare TaggedDocuments ===

# # train_tagged = [TaggedDocument(words=row['claim_text'], tags=[str(i)]) for i, row in train_df.iterrows()]
# # test_tagged = [TaggedDocument(words=row['claim_text'], tags=[str(i)]) for i, row in test_df.iterrows()]

# # # === Load Doc2Vec model ===
# # model = Doc2Vec.load('doc2vec.model')

# # # === Infer vectors ===
# # train_vectors = [model.infer_vector(doc.words) for doc in train_tagged]
# # train_labels = train_df['claim_label'].tolist()
# # test_vectors = [model.infer_vector(doc.words) for doc in test_tagged]

# # # === Encode labels ===
# # label_encoder = LabelEncoder()
# # train_labels_encoded = label_encoder.fit_transform(train_labels)

# # # === Train classifier ===
# # clf = LogisticRegression(max_iter=100)
# # clf.fit(train_vectors, train_labels_encoded)

# # # === Predict on test set ===
# # predicted_labels_encoded = clf.predict(test_vectors)
# # predicted_labels = label_encoder.inverse_transform(predicted_labels_encoded)

# # # === Prepare evidence vectors and texts ===
# # evidence_vectors = {row['key']: model.infer_vector(row['value']) for _, row in evidence_df.iterrows()}
# # evidence_texts = {row['key']: set(row['value']) for _, row in evidence_df.iterrows()}

# # # === Prepare original test claim text ===
# # test_df_og = pd.DataFrame(original_test_claims).transpose()
# # test_claim_text = test_df_og['claim_text'].apply(lambda x: ' '.join(x) if isinstance(x, list) else x)

# # # === Save predictions with evidence ===
# # output_dict = {}

# # for i, row in test_df.iterrows():
# #     claim_id = f"claim-{i}"
# #     original_text = test_claim_text[i]
# #     predicted_label = predicted_labels[i]
# #     claim_words = set(row['claim_text'])

# #     # --- Filter evidence by keyword overlap ---
# #     filtered_evidence_keys = [
# #         key for key, words in evidence_texts.items()
# #         if claim_words & words  # non-empty intersection
# #     ]

# #     if not filtered_evidence_keys:
# #         # fallback: use all evidence if no keyword match
# #         filtered_evidence_keys = list(evidence_vectors.keys())

# #     # --- Compute similarity only on filtered set ---
# #     filtered_evidence_matrix = np.array([evidence_vectors[k] for k in filtered_evidence_keys])
# #     claim_vector = model.infer_vector(row['claim_text'])
# #     similarities = cosine_similarity([claim_vector], filtered_evidence_matrix)[0]

# #     # --- Get top 3 evidence IDs ---
# #     top_indices = similarities.argsort()[-3:][::-1]
# #     top_evidence_ids = [filtered_evidence_keys[idx] for idx in top_indices]

# #     output_dict[claim_id] = {
# #         "claim_text": original_text,
# #         "claim_label": predicted_label,
# #         "evidences": top_evidence_ids
# #     }

# # # === Save JSON ===
# # with open('doc2vec_test_predictions_with_evidence.json', 'w') as f:
# #     json.dump(output_dict, f, indent=2)

# # print("✅ Saved predictions with supporting evidences to doc2vec_test_predictions_with_evidence.json")


# # import pandas as pd
# # import ast
# # import json
# # import numpy as np
# # from gensim.models.doc2vec import Doc2Vec, TaggedDocument
# # from sklearn.linear_model import LogisticRegression
# # from sklearn.preprocessing import LabelEncoder
# # from sklearn.metrics.pairwise import cosine_similarity

# # # Load original (raw) data
# # with open('data/train-claims.json', 'r') as f:
# #     raw_train_claims = json.load(f)

# # with open('data/dev-claims.json', 'r') as f:
# #     raw_dev_claims = json.load(f)

# # with open('data/dev-claims-baseline.json', 'r') as f:
# #     raw_dev_baseline_claims = json.load(f)

# # with open('data/test-claims-unlabelled.json', 'r') as f:
# #     raw_test_claims = json.load(f)

# # with open('data/evidence.json', 'r') as f:
# #     raw_evidence = json.load(f)


# # # load preprocessed data
# # train_df = pd.read_json('processed/preprocessed_train.json', orient='index')
# # dev_df = pd.read_json('processed/preprocessed_dev.json', orient='index')
# # dev_baseline_df = pd.read_json('processed/preprocessed_dev_baseline.json', orient='index')
# # test_df = pd.read_json('processed/preprocessed_test.json', orient='index')
# # evidence_df = pd.read_json('processed/preprocessed_evidence.json', orient='records')
# # retrieved_train_df = pd.read_json('processed/retrieved_train_evidence.json', orient='records')

# # print("Loaded preprocessed data...")
# # print(train_df.head())
# # print(dev_df.head())
# # print(dev_baseline_df.head())
# # print(test_df.head())
# # print(evidence_df.head())
# # print(retrieved_train_df.head())

# # with open('processed/retrieved_train_evidence.json', 'r') as f:
# #     retrieved_train_dict = json.load(f)

# # classification_data = []

# # for claim_id, row in train_df.iterrows():
# #     raw_claim = raw_train_claims[claim_id]['claim_text']
# #     label = raw_train_claims[claim_id]['claim_label']
# #     gold_evidence_ids = raw_train_claims[claim_id].get('evidences', [])
    
# #     if not gold_evidence_ids:
# #         continue

# #     # Get the gold evidence texts
# #     evidence_texts = [raw_evidence[eid] for eid in gold_evidence_ids if eid in raw_evidence]

# #     # Format input for classification
# #     combined_input = f"Claim: {raw_claim} "
# #     for i, et in enumerate(evidence_texts):
# #         combined_input += f"[SEP] Evidence {i+1}: {et} "

# #     classification_data.append({
# #         'claim_id': claim_id,
# #         'input_text': combined_input.strip(),
# #         'label': label,
# #         'evidences': gold_evidence_ids  # ✅ now using gold evidence
# #     })

# # classification_df = pd.DataFrame(classification_data)
# # print("Prepared training dataset with gold evidence:")
# # print(classification_df.head())

# ################################################################################
# import torch
# from transformers import pipeline

# # setting device on GPU if available, else CPU
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print('Using device:', device)
# print()

# if device.type == 'cuda':
#     print(torch.cuda.get_device_name(0))
#     print('Memory Usage:')
#     print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3, 1), 'GB')
#     print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3, 1), 'GB')

# # import transformers
# # print("Torch version:", torch.__version__)
# # print("Transformers version:", transformers.__version__)
# # print("Transformers detects framework:", transformers.file_utils.is_torch_available())

# # # Now the pipeline should work
# # classifier = pipeline(
# #     "zero-shot-classification"
# # )

# # result = classifier(
# #     "According to all known laws of aviation, there is no way that a bee should be able to fly. Its wings are too small to get its fat little body off the ground.",
# #     candidate_labels=["SUPPORTS", "REFUTES", "DISPUTED", "NOT_ENOUGH_INFO"]
# # )

# # print(result)

# import json
# import pandas as pd
# from transformers import pipeline

# # === Load data ===
# with open("processed/retrieved_train_evidence.json") as f:
#     claim_to_evid = json.load(f)

# with open("data/train-claims.json") as f:
#     claims_list = json.load(f)  # Should be a list of dicts

# with open("data/evidence.json") as f:
#     evidence_list = json.load(f)  # Should be a list of dicts


# # Convert to dicts for easy lookup
# claim_dict = {cid: cdata["claim_text"] for cid, cdata in claims_list.items()}
# evidence_dict = evidence_list 


# # === Set up pipeline ===
# classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# # === Process ===
# output = []

# output = []

# for claim_id, evid_ids in claim_to_evid.items():
#     print("#", end="", flush=True)

#     if claim_id not in claim_dict:
#         continue

#     claim = claim_dict[claim_id]
#     used_evidences = [evidence_dict[eid] for eid in evid_ids if eid in evidence_dict]

#     if not used_evidences:
#         continue

#     combined_evidence = " ".join(used_evidences)

#     result = classifier(
#         claim,
#         candidate_labels=["SUPPORTS", "REFUTES", "DISPUTED", "NOT_ENOUGH_INFO"],
#         hypothesis_template="{}"
#     )

#     output.append({
#         "claim_id": claim_id,
#         "claim": claim,
#         "predicted_label": result["labels"][0],
#         "evidences_used": used_evidences
#     })




# # === Output ===
# df_out = pd.DataFrame(output)
# df_out.to_csv("predicted_claims_with_evidence.csv", index=False)


import json
from transformers import pipeline

# === Load data ===
with open("data/test-claims-unlabelled.json") as f:
    test_claims = json.load(f)

with open("data/evidence.json") as f:
    raw_evidence_dict = json.load(f)

# === Normalize evidence keys (ensure they match "evidence-*" format) ===
evidence_dict = {
    f"evidence-{k}" if not k.startswith("evidence-") else k: v
    for k, v in raw_evidence_dict.items()
}

# === Set up classifier ===
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# === Predict labels ===
output = {}

for claim_id, claim_data in test_claims.items():
    print("#", end="", flush=True)

    claim_text = claim_data["claim_text"]
    evidence_ids = claim_data.get("evidences", [])

    print(f"\n🔍 {claim_id} evidence_ids: {evidence_ids[:3]}")  
    print(f"Sample evidence_dict keys: {list(evidence_dict.keys())[:3]}")
    used_evidences = [evidence_dict[eid] for eid in evidence_ids if eid in evidence_dict]

    if not used_evidences:
        continue  # skip if no matching evidence found

    combined_evidence = " ".join(used_evidences)

    result = classifier(
        claim_text,
        candidate_labels=["SUPPORTS", "REFUTES", "DISPUTED", "NOT_ENOUGH_INFO"],
        hypothesis_template="{}"
    )

    output[claim_id] = {
        "claim_text": claim_text,
        "predicted_label": result["labels"][0],
        "evidences": evidence_ids
    }

# === Save output ===
with open("predicted_test_claims_with_evidence.json", "w") as f:
    json.dump(output, f, indent=2)

print("\n✅ Saved predictions to predicted_test_claims_with_evidence.json")
