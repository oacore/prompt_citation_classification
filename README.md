# prompt_citation_classification

This repository contains the code for the CIKM 2023 paper, "Prompting Strategies for Citation Classification".

The code for pet is adapted from - [How Many Data Points is a Prompt Worth?](https://github.com/TevenLeScao/pet)

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
