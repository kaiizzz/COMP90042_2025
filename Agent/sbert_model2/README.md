---
tags:
- sentence-transformers
- sentence-similarity
- feature-extraction
- generated_from_trainer
- dataset_size:14761
- loss:MultipleNegativesRankingLoss
base_model: sentence-transformers/multi-qa-MiniLM-L6-cos-v1
widget:
- source_sentence: Most of the atmospheric moisture originates in the tropical ocean,
    and the difference between surface and upper atmospheric temperature determines
    how much of the moisture rises into the atmosphere.
  sentences:
  - The Kertha Gosa Pavilion at Klungkung has the story of Bhima Swarga painted around
    the ceiling.
  - Hawaii is a state of the United States, nearly coterminous with the Hawaiian Islands.
  - Donald MacLennan (March 22, 1875 -- October 19, 1953) was a lawyer and political
    figure in Nova Scotia, Canada.
- source_sentence: Ancient natural cycles are irrelevant for attributing recent global
    warming to humans.
  sentences:
  - It lies in between Gudiyattam to Chittoor State High Way.
  - John 6:10 αναπεσον ] αναπεσαν
  - The area of the sheet that experiences melting has been argued to have increased
    by about 16% between 1979 (when measurements started) and 2002 (most recent data).
- source_sentence: '"The climate of this planet oscillates between periods of approximately  30
    years of warming followed by approximately 30 years of cooling.'
  sentences:
  - In 2012, James rose to prominence after winning the nineteenth cycle of America
    's Next Top Model, and was consequentially signed with L.A. Models and New York
    Model Management.
  - Carbon dioxide is colorless.
  - 'Bhookamp (English : Earthquake) is a 1993 Indian Bollywood Action film produced
    by Markand Adhikari on Sri Adhikari Brothers banner and directed by Gautam Adhikari.'
- source_sentence: Julia Gillard her decision not to argue against a fixed carbon
    price being labelled a "carbon tax" hurt her terribly politically.
  sentences:
  - The Idaho Department of Fish and Game acquired the land in 1987 with the help
    of Ducks Unlimited and The Nature Conservancy to provide quality wetland and upland
    habitat for migratory and resident wildlife.
  - The 29th Armoured Brigade was a Second World War British Army brigade.
  - The range of natural distribution is from the Nambucca River, New South Wales
    to Iron Range National Park, in north Queensland.
- source_sentence: U.S. Pays $1 Billion into Green Climate Fund, Top Polluters Pay
    Nothing
  sentences:
  - 'Category : Bharatiya Mazdoor Sangh-affiliated unions'
  - In their 20th year under head coach Henry L. Williams, the Golden Gophers compiled
    a 4-2-1 record (3 -- 2 against Big Ten Conference opponents).
  - Non-binding referendums regarding Puerto Rico 's status have been held in 1967,
    1993, 1997 and 2012.
pipeline_tag: sentence-similarity
library_name: sentence-transformers
---

# SentenceTransformer based on sentence-transformers/multi-qa-MiniLM-L6-cos-v1

This is a [sentence-transformers](https://www.SBERT.net) model finetuned from [sentence-transformers/multi-qa-MiniLM-L6-cos-v1](https://huggingface.co/sentence-transformers/multi-qa-MiniLM-L6-cos-v1). It maps sentences & paragraphs to a 384-dimensional dense vector space and can be used for semantic textual similarity, semantic search, paraphrase mining, text classification, clustering, and more.

## Model Details

### Model Description
- **Model Type:** Sentence Transformer
- **Base model:** [sentence-transformers/multi-qa-MiniLM-L6-cos-v1](https://huggingface.co/sentence-transformers/multi-qa-MiniLM-L6-cos-v1) <!-- at revision b207367332321f8e44f96e224ef15bc607f4dbf0 -->
- **Maximum Sequence Length:** 512 tokens
- **Output Dimensionality:** 384 dimensions
- **Similarity Function:** Cosine Similarity
<!-- - **Training Dataset:** Unknown -->
<!-- - **Language:** Unknown -->
<!-- - **License:** Unknown -->

### Model Sources

- **Documentation:** [Sentence Transformers Documentation](https://sbert.net)
- **Repository:** [Sentence Transformers on GitHub](https://github.com/UKPLab/sentence-transformers)
- **Hugging Face:** [Sentence Transformers on Hugging Face](https://huggingface.co/models?library=sentence-transformers)

### Full Model Architecture

```
SentenceTransformer(
  (0): Transformer({'max_seq_length': 512, 'do_lower_case': False}) with Transformer model: BertModel 
  (1): Pooling({'word_embedding_dimension': 384, 'pooling_mode_cls_token': False, 'pooling_mode_mean_tokens': True, 'pooling_mode_max_tokens': False, 'pooling_mode_mean_sqrt_len_tokens': False, 'pooling_mode_weightedmean_tokens': False, 'pooling_mode_lasttoken': False, 'include_prompt': True})
  (2): Normalize()
)
```

## Usage

### Direct Usage (Sentence Transformers)

First install the Sentence Transformers library:

```bash
pip install -U sentence-transformers
```

Then you can load this model and run inference.
```python
from sentence_transformers import SentenceTransformer

# Download from the 🤗 Hub
model = SentenceTransformer("sentence_transformers_model_id")
# Run inference
sentences = [
    'U.S. Pays $1 Billion into Green Climate Fund, Top Polluters Pay Nothing',
    'Category : Bharatiya Mazdoor Sangh-affiliated unions',
    'In their 20th year under head coach Henry L. Williams, the Golden Gophers compiled a 4-2-1 record (3 -- 2 against Big Ten Conference opponents).',
]
embeddings = model.encode(sentences)
print(embeddings.shape)
# [3, 384]

# Get the similarity scores for the embeddings
similarities = model.similarity(embeddings, embeddings)
print(similarities.shape)
# [3, 3]
```

<!--
### Direct Usage (Transformers)

<details><summary>Click to see the direct usage in Transformers</summary>

</details>
-->

<!--
### Downstream Usage (Sentence Transformers)

You can finetune this model on your own dataset.

<details><summary>Click to expand</summary>

</details>
-->

<!--
### Out-of-Scope Use

*List how the model may foreseeably be misused and address what users ought not to do with the model.*
-->

<!--
## Bias, Risks and Limitations

*What are the known or foreseeable issues stemming from this model? You could also flag here known failure cases or weaknesses of the model.*
-->

<!--
### Recommendations

*What are recommendations with respect to the foreseeable issues? For example, filtering explicit content.*
-->

## Training Details

### Training Dataset

#### Unnamed Dataset

* Size: 14,761 training samples
* Columns: <code>claim</code>, <code>evidence</code>, and <code>label</code>
* Approximate statistics based on the first 1000 samples:
  |         | claim                                                                             | evidence                                                                           | label                                            |
  |:--------|:----------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------|:-------------------------------------------------|
  | type    | string                                                                            | string                                                                             | int                                              |
  | details | <ul><li>min: 7 tokens</li><li>mean: 26.46 tokens</li><li>max: 73 tokens</li></ul> | <ul><li>min: 8 tokens</li><li>mean: 30.96 tokens</li><li>max: 107 tokens</li></ul> | <ul><li>-1: ~72.90%</li><li>1: ~27.10%</li></ul> |
* Samples:
  | claim                                                                                                                                                                                                                                                                                                                                        | evidence                                                                                                                                                                                                                                 | label           |
  |:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:----------------|
  | <code>Great Barrier Reef may perish by 2030s</code>                                                                                                                                                                                                                                                                                          | <code>The other OEF mission in Africa is known as Operation Enduring Freedom -- Trans Sahara (OEF-TS), which, until the creation of the new United States Africa Command, was run from the United States European Command.</code>        | <code>-1</code> |
  | <code>They concluded that trends toward rising climate damages were mainly due to increased population and economic activity in the path of storms, that it was not currently possible to determine the portion of damages attributable to greenhouse gases, and that they didn’t expect that situation to change in the near future.</code> | <code>The initial version of the software was written by Dr. Bob MacMillan as part of his Ph.D. thesis at the University of Edinburgh in 1993, and was coded in a Microsoft Rapid Application Development language called FoxPro.</code> | <code>-1</code> |
  | <code>Some of the regions in which  GRACE claims ice loss in East Antarctica average colder than -30°C  during the summer, and never, ever get above freezing.</code>                                                                                                                                                                        | <code>James Anderson (May 28, 1849 -- May 31, 1918), born James Anderson Smythe, was a Canadian-born soldier in the U.S. Army who served with the 6th U.S. Cavalry during the Texas -- Indian Wars.</code>                               | <code>-1</code> |
* Loss: [<code>MultipleNegativesRankingLoss</code>](https://sbert.net/docs/package_reference/sentence_transformer/losses.html#multiplenegativesrankingloss) with these parameters:
  ```json
  {
      "scale": 20.0,
      "similarity_fct": "cos_sim"
  }
  ```

### Evaluation Dataset

#### Unnamed Dataset

* Size: 1,641 evaluation samples
* Columns: <code>claim</code>, <code>evidence</code>, and <code>label</code>
* Approximate statistics based on the first 1000 samples:
  |         | claim                                                                             | evidence                                                                           | label                                            |
  |:--------|:----------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------|:-------------------------------------------------|
  | type    | string                                                                            | string                                                                             | int                                              |
  | details | <ul><li>min: 7 tokens</li><li>mean: 26.86 tokens</li><li>max: 66 tokens</li></ul> | <ul><li>min: 6 tokens</li><li>mean: 30.79 tokens</li><li>max: 147 tokens</li></ul> | <ul><li>-1: ~73.80%</li><li>1: ~26.20%</li></ul> |
* Samples:
  | claim                                                                                                                                              | evidence                                                                                                                                                                                                                                                     | label           |
  |:---------------------------------------------------------------------------------------------------------------------------------------------------|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:----------------|
  | <code>Sea level rise is not going to happen.</code>                                                                                                | <code>In the field of hyperbolic geometry, the order-4 hexagonal tiling honeycomb arises as one of 11 regular paracompact honeycombs in 3-dimensional hyperbolic space.</code>                                                                               | <code>-1</code> |
  | <code>The main greenhouse gas is water vapour[…]</code>                                                                                            | <code>The latter case is not to be confused with the passive voice, where only the direct object of a sentence becomes the subject of the passive-voiced sentence, and the verb 's structure also changes to convey the meaning of the passive voice.</code> | <code>-1</code> |
  | <code>"[Models] are full of fudge factors that are fitted to the existing climate, so the models more or less agree with the observed data.</code> | <code>First Russell ministry, the British government led by Lord John Russell from 1846 to 1852</code>                                                                                                                                                       | <code>-1</code> |
* Loss: [<code>MultipleNegativesRankingLoss</code>](https://sbert.net/docs/package_reference/sentence_transformer/losses.html#multiplenegativesrankingloss) with these parameters:
  ```json
  {
      "scale": 20.0,
      "similarity_fct": "cos_sim"
  }
  ```

### Training Hyperparameters
#### Non-Default Hyperparameters

- `eval_strategy`: steps
- `per_device_train_batch_size`: 16
- `per_device_eval_batch_size`: 16
- `learning_rate`: 2e-05
- `warmup_ratio`: 0.1
- `fp16`: True

#### All Hyperparameters
<details><summary>Click to expand</summary>

- `overwrite_output_dir`: False
- `do_predict`: False
- `eval_strategy`: steps
- `prediction_loss_only`: True
- `per_device_train_batch_size`: 16
- `per_device_eval_batch_size`: 16
- `per_gpu_train_batch_size`: None
- `per_gpu_eval_batch_size`: None
- `gradient_accumulation_steps`: 1
- `eval_accumulation_steps`: None
- `torch_empty_cache_steps`: None
- `learning_rate`: 2e-05
- `weight_decay`: 0.0
- `adam_beta1`: 0.9
- `adam_beta2`: 0.999
- `adam_epsilon`: 1e-08
- `max_grad_norm`: 1.0
- `num_train_epochs`: 3
- `max_steps`: -1
- `lr_scheduler_type`: linear
- `lr_scheduler_kwargs`: {}
- `warmup_ratio`: 0.1
- `warmup_steps`: 0
- `log_level`: passive
- `log_level_replica`: warning
- `log_on_each_node`: True
- `logging_nan_inf_filter`: True
- `save_safetensors`: True
- `save_on_each_node`: False
- `save_only_model`: False
- `restore_callback_states_from_checkpoint`: False
- `no_cuda`: False
- `use_cpu`: False
- `use_mps_device`: False
- `seed`: 42
- `data_seed`: None
- `jit_mode_eval`: False
- `use_ipex`: False
- `bf16`: False
- `fp16`: True
- `fp16_opt_level`: O1
- `half_precision_backend`: auto
- `bf16_full_eval`: False
- `fp16_full_eval`: False
- `tf32`: None
- `local_rank`: 0
- `ddp_backend`: None
- `tpu_num_cores`: None
- `tpu_metrics_debug`: False
- `debug`: []
- `dataloader_drop_last`: False
- `dataloader_num_workers`: 0
- `dataloader_prefetch_factor`: None
- `past_index`: -1
- `disable_tqdm`: False
- `remove_unused_columns`: True
- `label_names`: None
- `load_best_model_at_end`: False
- `ignore_data_skip`: False
- `fsdp`: []
- `fsdp_min_num_params`: 0
- `fsdp_config`: {'min_num_params': 0, 'xla': False, 'xla_fsdp_v2': False, 'xla_fsdp_grad_ckpt': False}
- `tp_size`: 0
- `fsdp_transformer_layer_cls_to_wrap`: None
- `accelerator_config`: {'split_batches': False, 'dispatch_batches': None, 'even_batches': True, 'use_seedable_sampler': True, 'non_blocking': False, 'gradient_accumulation_kwargs': None}
- `deepspeed`: None
- `label_smoothing_factor`: 0.0
- `optim`: adamw_torch
- `optim_args`: None
- `adafactor`: False
- `group_by_length`: False
- `length_column_name`: length
- `ddp_find_unused_parameters`: None
- `ddp_bucket_cap_mb`: None
- `ddp_broadcast_buffers`: False
- `dataloader_pin_memory`: True
- `dataloader_persistent_workers`: False
- `skip_memory_metrics`: True
- `use_legacy_prediction_loop`: False
- `push_to_hub`: False
- `resume_from_checkpoint`: None
- `hub_model_id`: None
- `hub_strategy`: every_save
- `hub_private_repo`: None
- `hub_always_push`: False
- `gradient_checkpointing`: False
- `gradient_checkpointing_kwargs`: None
- `include_inputs_for_metrics`: False
- `include_for_metrics`: []
- `eval_do_concat_batches`: True
- `fp16_backend`: auto
- `push_to_hub_model_id`: None
- `push_to_hub_organization`: None
- `mp_parameters`: 
- `auto_find_batch_size`: False
- `full_determinism`: False
- `torchdynamo`: None
- `ray_scope`: last
- `ddp_timeout`: 1800
- `torch_compile`: False
- `torch_compile_backend`: None
- `torch_compile_mode`: None
- `include_tokens_per_second`: False
- `include_num_input_tokens_seen`: False
- `neftune_noise_alpha`: None
- `optim_target_modules`: None
- `batch_eval_metrics`: False
- `eval_on_start`: False
- `use_liger_kernel`: False
- `eval_use_gather_object`: False
- `average_tokens_across_devices`: False
- `prompts`: None
- `batch_sampler`: batch_sampler
- `multi_dataset_batch_sampler`: proportional

</details>

### Training Logs
| Epoch  | Step | Training Loss | Validation Loss |
|:------:|:----:|:-------------:|:---------------:|
| 0.1083 | 100  | 4.3762        | 3.1636          |
| 0.2167 | 200  | 3.0097        | 2.8153          |
| 0.3250 | 300  | 2.7236        | 2.4919          |
| 0.4334 | 400  | 2.5135        | 2.4065          |
| 0.5417 | 500  | 2.4476        | 2.3767          |
| 0.6501 | 600  | 2.3958        | 2.3711          |
| 0.7584 | 700  | 2.399         | 2.3266          |
| 0.8667 | 800  | 2.3507        | 2.3288          |
| 0.9751 | 900  | 2.3693        | 2.3100          |
| 1.0834 | 1000 | 2.2346        | 2.3039          |
| 1.1918 | 1100 | 2.2437        | 2.3309          |
| 1.3001 | 1200 | 2.2374        | 2.3190          |
| 1.4085 | 1300 | 2.2477        | 2.3109          |
| 1.5168 | 1400 | 2.2479        | 2.3032          |
| 1.6251 | 1500 | 2.2603        | 2.2893          |
| 1.7335 | 1600 | 2.1977        | 2.2930          |
| 1.8418 | 1700 | 2.2627        | 2.2771          |
| 1.9502 | 1800 | 2.1952        | 2.2881          |
| 2.0585 | 1900 | 2.1439        | 2.3070          |
| 2.1668 | 2000 | 2.1643        | 2.3056          |
| 2.2752 | 2100 | 2.0675        | 2.3175          |
| 2.3835 | 2200 | 2.0611        | 2.3193          |
| 2.4919 | 2300 | 2.0787        | 2.3117          |
| 2.6002 | 2400 | 2.0805        | 2.3097          |
| 2.7086 | 2500 | 2.097         | 2.3057          |
| 2.8169 | 2600 | 2.1547        | 2.3121          |
| 2.9252 | 2700 | 2.0876        | 2.3126          |


### Framework Versions
- Python: 3.10.11
- Sentence Transformers: 4.1.0
- Transformers: 4.51.3
- PyTorch: 2.7.0+cu128
- Accelerate: 1.6.0
- Datasets: 3.6.0
- Tokenizers: 0.21.1

## Citation

### BibTeX

#### Sentence Transformers
```bibtex
@inproceedings{reimers-2019-sentence-bert,
    title = "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",
    author = "Reimers, Nils and Gurevych, Iryna",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing",
    month = "11",
    year = "2019",
    publisher = "Association for Computational Linguistics",
    url = "https://arxiv.org/abs/1908.10084",
}
```

#### MultipleNegativesRankingLoss
```bibtex
@misc{henderson2017efficient,
    title={Efficient Natural Language Response Suggestion for Smart Reply},
    author={Matthew Henderson and Rami Al-Rfou and Brian Strope and Yun-hsuan Sung and Laszlo Lukacs and Ruiqi Guo and Sanjiv Kumar and Balint Miklos and Ray Kurzweil},
    year={2017},
    eprint={1705.00652},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
```

<!--
## Glossary

*Clearly define terms in order to be accessible across audiences.*
-->

<!--
## Model Card Authors

*Lists the people who create the model card, providing recognition and accountability for the detailed work that goes into its construction.*
-->

<!--
## Model Card Contact

*Provides a way for people who have updates to the Model Card, suggestions, or questions, to contact the Model Card authors.*
-->