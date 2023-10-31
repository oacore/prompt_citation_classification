# prompt_citation_classification

This repository contains the code for the CIKM 2023 paper, [Prompting Strategies for Citation Classification](https://dl.acm.org/doi/pdf/10.1145/3583780.3615018).

The code for PET is adapted from - [How Many Data Points is a Prompt Worth?](https://github.com/TevenLeScao/pet)

For the fixed prompt LM tuning and dynamic context prompt LM tuning:

```

!python cli.py \
--method pet \
--pattern_ids ${pid} \
--data_type ${data_type} \
--data_dir data/ \
--model_type scibert \
--model_name_or_path allenai/scibert_scivocab_uncased \
--task_name function \
--output_dir output_dir/ \
--do_train \
--do_eval \
--do_test \
--pet_per_gpu_eval_batch_size 4 \
--pet_per_gpu_train_batch_size 4 \
--pet_gradient_accumulation_steps 4 \
--pet_num_train_epochs 5 \
--pet_min_steps 250 \
--pet_max_steps 2000 \
--pet_max_seq_length 256 \
--pet_repetitions 4 \
--sc_per_gpu_train_batch_size 4 \
--sc_per_gpu_unlabeled_batch_size 16 \
--sc_gradient_accumulation_steps 4 \
--sc_num_train_epochs 5 \
--sc_min_steps 250 \
--sc_max_seq_length 256 \
--sc_repetitions 1 \
--train_examples 10 50 100 500 1000 1372 \
--warmup_steps 50 \
--logging_steps 50 \
--overwrite_output_dir \
--no_distillation

```

For promptless fine-tuning:

```
!python cli.py \
--method sequence_classifier \
--data_dir data/ \
--model_type scibert \
--model_name_or_path allenai/scibert_scivocab_uncased \
--task_name function \
--output_dir output/ \
--do_train \
--do_eval \
--do_test \
--sc_per_gpu_train_batch_size 4 \
--sc_per_gpu_eval_batch_size 4 \
--sc_per_gpu_unlabeled_batch_size 4 \
--sc_gradient_accumulation_steps 4 \
--sc_min_steps 250 \
--sc_num_train_epochs 4 \
--sc_max_seq_length 256 \
--sc_repetitions 4 \
--train_examples 10 50 100 500 1000 1372 \
--warmup_steps 50 \
--logging_steps 50 \
--overwrite_output_dir \
--no_distillation \
```

For zero-shot tuning free prompting:

```
!python tuning_free/main.py \
  --method "zero_shot_reason" \
  --input_file 'test.txt' \
  --model 'gpt3.5' \
  --output_dir output/ \
  --dataset ${dataset_name}
```
Code for tuning free prompting is based on [Large Language Models are Zero-Shot Reasoners](https://github.com/kojima-takeshi188/zero_shot_cot)

For datasets (ACL-ARC and ACT2) and code for dynamic citation context extraction, checkout the code here - [Dynamic Context Extraction for Citation Classification](https://github.com/oacore/dynamic_citation_context)

### Citation

Please cite our paper as follows:
```
@inproceedings{10.1145/3583780.3615018,
author = {Kunnath, Suchetha N. and Pride, David and Knoth, Petr},
title = {Prompting Strategies for Citation Classification},
year = {2023},
isbn = {9798400701245},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3583780.3615018},
doi = {10.1145/3583780.3615018},
abstract = {Citation classification aims to identify the purpose of the cited article in the citing article. Previous citation classification methods rely largely on supervised approaches. The models are trained on datasets with citing sentences or citation contexts annotated for a citation's purpose or function or intent. Recent advancements in Large Language Models (LLMs) have dramatically improved the ability of NLP systems to achieve state-of-the-art performances under zero or few-shot settings. This makes LLMs particularly suitable for tasks where sufficiently large labelled datasets are not yet available, which remains to be the case for citation classification. This paper systematically investigates the effectiveness of different prompting strategies for citation classification and compares them to promptless strategies as a baseline. Specifically, we evaluate the following four strategies, two of which we introduce for the first time, which involve updating Language Model (LM) parameters while training the model: (1) Promptless fine-tuning, (2) Fixed-prompt LM tuning, (3) Dynamic Context-prompt LM tuning (proposed), (4) Prompt + LM fine-tuning (proposed). Additionally, we test the zero-shot performance of LLMs, GPT3.5, a (5) Tuning-free prompting strategy that involves no parameter updating. Our results show that prompting methods based on LM parameter updating significantly improve citation classification performances on both domain-specific and multi-disciplinary citation classifications. Moreover, our Dynamic Context-prompting method achieves top scores both for the ACL-ARC and ACT2 citation classification datasets, surpassing the highest-performing system in the 3C shared task benchmark. Interestingly, we observe zero-shot GPT3.5 to perform well on ACT2 but poorly on the ACL-ARC dataset.},
booktitle = {Proceedings of the 32nd ACM International Conference on Information and Knowledge Management},
pages = {1127–1137},
numpages = {11},
keywords = {prompt training, research evaluation, large language models, citation classification},
location = {Birmingham, United Kingdom},
series = {CIKM '23}
}
```
