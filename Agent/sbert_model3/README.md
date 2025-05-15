---
tags:
- sentence-transformers
- sentence-similarity
- feature-extraction
- generated_from_trainer
- dataset_size:14761
- loss:CachedMultipleNegativesRankingLoss
base_model: sentence-transformers/all-MiniLM-L6-v2
widget:
- source_sentence: Most of the atmospheric moisture originates in the tropical ocean,
    and the difference between surface and upper atmospheric temperature determines
    how much of the moisture rises into the atmosphere.
  sentences:
  - Some of this stems from difficulties programming older traffic signal control
    software to prevent the yellow trap, but much stems from traffic engineers or
    technicians not understanding the yellow trap hazard, or believing it is not a
    serious problem.
  - He won the Scottish Mini Cooper Cup three times and the Mini Challenge UK series
    twice.
  - It is also sold as a prepaid bundle for less than $ 100 at most retailers.
- source_sentence: Renewables can't provide baseload power.
  sentences:
  - Petersburg High School (PHS) is the public high school for the Southeast Alaskan
    community of Petersburg and the Petersburg City School District.
  - TAA stands for Trans Australia Airlines.
  - At the same time, the solar power industry grew by almost a quarter to 374,000
    jobs.
- source_sentence: They concluded that trends toward rising climate damages were mainly
    due to increased population and economic activity in the path of storms, that
    it was not currently possible to determine the portion of damages attributable
    to greenhouse gases, and that they didn’t expect that situation to change in the
    near future.
  sentences:
  - The previous record of the lowest area of the Arctic Ocean covered by ice in 2012
    saw a low of 1.58 million square miles (4.09 million square kilometers).
  - Some of the meteorological variables that are commonly measured are temperature,
    humidity, atmospheric pressure, wind, and precipitation.
  - Lars Benzon (1687 -- 1742) was a landowner and a deputy (deputeret) in the commissariat
    generalof the Danish navy.
- source_sentence: The Schmittner et al. study finds low probability of both very
    low and very high climate sensitivities, and its lower estimate (as compared to
    the IPCC) is based on a new temperature reconstruction of the Last Glacial Maximum
    that may or may not withstand the test of time.
  sentences:
  - She has appeared as soloist at the Concertgebouw Amsterdam, the Gewandhausorchester
    Leipzig, the Staatskapelle Dresden, the Orchestre de Paris, the Moscow Philharmonic
    Orchestra and the Saint Petersburg Philharmonic Orchestra, with conductors including
    Claudio Abbado, Karl Böhm, Aleksandr Dmitriyev, Valery Gergiev, Neeme Järvi, Kirill
    Kondrashin, Kurt Masur and Yehudi Menuhin.
  - The Medieval Warm Period (MWP) also known as the Medieval Climate Optimum, or
    Medieval Climatic Anomaly was a time of warm climate in the North Atlantic region
    lasting from c. 950 to c. 1250.
  - He was one of the founders of popular science movement in Kerala State, India.
- source_sentence: Scientists have determined that the factors which caused the Little
    Ice Age cooling are not currently causing global warming.
  sentences:
  - Its broadcast area reaches into southeastern Kentucky, southwestern Virginia,
    western North Carolina, north Georgia and far northwest South Carolina.
  - The Phu Ruea High Altitude Agricultural Research Station and the Phu Ruea National
    Park are located in the area around the mountain.
  - Researchers have for the first time attributed recent floods, droughts and heat
    waves, to human-induced climate change.
pipeline_tag: sentence-similarity
library_name: sentence-transformers
---

# SentenceTransformer based on sentence-transformers/all-MiniLM-L6-v2

This is a [sentence-transformers](https://www.SBERT.net) model finetuned from [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2). It maps sentences & paragraphs to a 384-dimensional dense vector space and can be used for semantic textual similarity, semantic search, paraphrase mining, text classification, clustering, and more.

## Model Details

### Model Description
- **Model Type:** Sentence Transformer
- **Base model:** [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) <!-- at revision c9745ed1d9f207416be6d2e6f8de32d1f16199bf -->
- **Maximum Sequence Length:** 256 tokens
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
  (0): Transformer({'max_seq_length': 256, 'do_lower_case': False}) with Transformer model: BertModel 
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
    'Scientists have determined that the factors which caused the Little Ice Age cooling are not currently causing global warming.',
    'Its broadcast area reaches into southeastern Kentucky, southwestern Virginia, western North Carolina, north Georgia and far northwest South Carolina.',
    'The Phu Ruea High Altitude Agricultural Research Station and the Phu Ruea National Park are located in the area around the mountain.',
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
  | details | <ul><li>min: 7 tokens</li><li>mean: 26.46 tokens</li><li>max: 73 tokens</li></ul> | <ul><li>min: 5 tokens</li><li>mean: 31.48 tokens</li><li>max: 256 tokens</li></ul> | <ul><li>-1: ~72.90%</li><li>1: ~27.10%</li></ul> |
* Samples:
  | claim                                                                                                                                                                                                                                                                                                                                        | evidence                                                                                                                                                                                                                    | label           |
  |:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:----------------|
  | <code>Great Barrier Reef may perish by 2030s</code>                                                                                                                                                                                                                                                                                          | <code>The Cox -- Forbes theory is a long-debunked theory on the evolution of chess put forward by Captain Hiram Cox (1760 -- 1799) and extended by Professor Duncan Forbes (1798 -- 1868).</code>                           | <code>-1</code> |
  | <code>They concluded that trends toward rising climate damages were mainly due to increased population and economic activity in the path of storms, that it was not currently possible to determine the portion of damages attributable to greenhouse gases, and that they didn’t expect that situation to change in the near future.</code> | <code>Charles James Lyall (1845 -- 1920), English civil servant working in India</code>                                                                                                                                     | <code>-1</code> |
  | <code>Some of the regions in which  GRACE claims ice loss in East Antarctica average colder than -30°C  during the summer, and never, ever get above freezing.</code>                                                                                                                                                                        | <code>The News-Herald is a newspaper distributed in the northeastern portion of Greater Cleveland, Ohio, United States, serving Lake, Geauga and Ashtabula Counties as well as a section of eastern Cuyahoga County.</code> | <code>-1</code> |
* Loss: [<code>CachedMultipleNegativesRankingLoss</code>](https://sbert.net/docs/package_reference/sentence_transformer/losses.html#cachedmultiplenegativesrankingloss) with these parameters:
  ```json
  {
      "scale": 20.0,
      "similarity_fct": "cos_sim",
      "mini_batch_size": 32
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
  | details | <ul><li>min: 7 tokens</li><li>mean: 26.86 tokens</li><li>max: 66 tokens</li></ul> | <ul><li>min: 6 tokens</li><li>mean: 30.39 tokens</li><li>max: 132 tokens</li></ul> | <ul><li>-1: ~73.80%</li><li>1: ~26.20%</li></ul> |
* Samples:
  | claim                                                                                                                                              | evidence                                                                                                                                                                               | label           |
  |:---------------------------------------------------------------------------------------------------------------------------------------------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:----------------|
  | <code>Sea level rise is not going to happen.</code>                                                                                                | <code>In August 2014, a rotational light-curve for this asteroid was obtained from photometric observations by Italian astronomers at the Franco Fuligni Observatory near Rome.</code> | <code>-1</code> |
  | <code>The main greenhouse gas is water vapour[…]</code>                                                                                            | <code>DEKA (New Zealand), a defunct discount store chain, formerly in New Zealand</code>                                                                                               | <code>-1</code> |
  | <code>"[Models] are full of fudge factors that are fitted to the existing climate, so the models more or less agree with the observed data.</code> | <code>Category : Populated places in the Province of Salamanca</code>                                                                                                                  | <code>-1</code> |
* Loss: [<code>CachedMultipleNegativesRankingLoss</code>](https://sbert.net/docs/package_reference/sentence_transformer/losses.html#cachedmultiplenegativesrankingloss) with these parameters:
  ```json
  {
      "scale": 20.0,
      "similarity_fct": "cos_sim",
      "mini_batch_size": 32
  }
  ```

### Training Hyperparameters
#### Non-Default Hyperparameters

- `eval_strategy`: steps
- `per_device_train_batch_size`: 32
- `per_device_eval_batch_size`: 32
- `learning_rate`: 0.02
- `num_train_epochs`: 0.5
- `warmup_ratio`: 0.1
- `fp16`: True

#### All Hyperparameters
<details><summary>Click to expand</summary>

- `overwrite_output_dir`: False
- `do_predict`: False
- `eval_strategy`: steps
- `prediction_loss_only`: True
- `per_device_train_batch_size`: 32
- `per_device_eval_batch_size`: 32
- `per_gpu_train_batch_size`: None
- `per_gpu_eval_batch_size`: None
- `gradient_accumulation_steps`: 1
- `eval_accumulation_steps`: None
- `torch_empty_cache_steps`: None
- `learning_rate`: 0.02
- `weight_decay`: 0.0
- `adam_beta1`: 0.9
- `adam_beta2`: 0.999
- `adam_epsilon`: 1e-08
- `max_grad_norm`: 1.0
- `num_train_epochs`: 0.5
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
| 0.2165 | 100  | 3.6111        | 3.4588          |
| 0.4329 | 200  | 3.4661        | 3.4588          |


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

#### CachedMultipleNegativesRankingLoss
```bibtex
@misc{gao2021scaling,
    title={Scaling Deep Contrastive Learning Batch Size under Memory Limited Setup},
    author={Luyu Gao and Yunyi Zhang and Jiawei Han and Jamie Callan},
    year={2021},
    eprint={2101.06983},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
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