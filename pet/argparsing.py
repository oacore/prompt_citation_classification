import argparse

from pet.tasks import PROCESSORS
from pet.wrapper import MODEL_CLASSES, WRAPPER_TYPES

# fmt: off
parser = argparse.ArgumentParser(description="Command line interface for PET/iPET")

# Required parameters
parser.add_argument("--method", required=True, choices=["pet", "ipet", "sequence_classifier"],
                    help="The training method to use. Either regular sequence classification, PET or iPET.", )
parser.add_argument("--data_dir", default=None, type=str, required=True,
                    help="The input data dir. Should contain the data files for the task.", )
parser.add_argument("--data_type", default=None, type=str, required=True,
                    help="The input data type. Options available are act2 and acl_arc", )                  
parser.add_argument("--model_type", default=None, type=str, required=True, choices=MODEL_CLASSES.keys(),
                    help="The type of the pretrained language model to use", )
parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
                    help="Path to the pre-trained model or shortcut name", )
parser.add_argument("--task_name", default=None, type=str, required=True, choices=PROCESSORS.keys(),
                    help="The name of the task to train/evaluate on", )
parser.add_argument("--output_dir", default=None, type=str, required=True,
                    help="The output directory where the model predictions and checkpoints will be written. "
                         "[TASK_NAME] will be replaced by the task name.", )
parser.add_argument("--save_model", action="store_true",
                    help="Whether to save model binaries. Keep disa bled to save space.")
parser.add_argument("--local_rank", default=-1, type=int)

# PET-specific optional parameters
parser.add_argument("--wrapper_type", default="mlm", choices=WRAPPER_TYPES,
                    help="The wrapper type. Set this to 'mlm' for a masked language model like BERT or to 'plm for a "
                         "permuted language model like XLNet (only for PET)", )
parser.add_argument("--pattern_ids", default=[0], type=int, nargs="+",
                    help="The ids of the PVPs to be used (only for PET)")
parser.add_argument("--lm_training", action="store_true",
                    help="Whether to use language modeling as auxiliary task (only for PET)")
parser.add_argument("--alpha", default=0.9999, type=float,
                    help="Weighting term for the auxiliary language modeling task (only for PET)", )
parser.add_argument("--temperature", default=2, type=float, help="Temperature used for combining PVPs (only for PET)")
parser.add_argument("--verbalizer_file", default=None,
                    help="The path to a file to override default verbalizers (only for PET)")
parser.add_argument("--reduction", default="wmean", choices=["wmean", "mean"],
                    help="Reduction strategy for merging predictions from multiple PET models. Select either uniform"
                         " weighting (mean) or weighting based on train set accuracy (wmean)", )
parser.add_argument("--decoding_strategy", default="default", choices=["default", "ltr", "parallel"],
                    help="The decoding strategy for PET with multiple masks (only for PET)", )
parser.add_argument("--no_distillation", action="store_true",
                    help="If set to true, no distillation is performed (only for PET)")
parser.add_argument("--pet_repetitions", default=3, type=int,
                    help="The number of times to repeat PET training and testing with different seeds.", )
parser.add_argument("--pet_max_seq_length", default=512, type=int,
                    help="The maximum total input sequence length after tokenization for PET. Sequences longer than "
                         "this will be truncated, sequences shorter will be padded.", )
parser.add_argument("--pet_per_gpu_train_batch_size", default=4, type=int,
                    help="Batch size per GPU/CPU for PET training.")
parser.add_argument("--pet_per_gpu_eval_batch_size", default=4, type=int,
                    help="Batch size per GPU/CPU for PET evaluation.")
parser.add_argument("--pet_per_gpu_unlabeled_batch_size", default=4, type=int,
                    help="Batch size per GPU/CPU for auxiliary language modeling examples in PET.", )
parser.add_argument("--pet_gradient_accumulation_steps", type=int, default=1,
                    help="Number of updates steps to accumulate before performing a backward/update pass in PET.", )
parser.add_argument("--pet_num_train_epochs", default=3, type=float,
                    help="Total number of training epochs to perform in PET.")
parser.add_argument("--pet_max_steps", default=-1, type=int,
                    help="If > 0: set total number of training steps to perform in PET. Override num_train_epochs.", )
parser.add_argument("--pet_min_steps", default=-1, type=int,
                    help="If > 0: set minimal number of steps. Use with num_train_epochs.")

# SequenceClassifier-specific optional parameters (also used for the final PET classifier)
parser.add_argument("--sc_repetitions", default=1, type=int,
                    help="The number of times to repeat seq. classifier training and testing with different seeds.", )
parser.add_argument("--sc_max_seq_length", default=512, type=int,
                    help="The maximum total input sequence length after tokenization for sequence classification. "
                         "Sequences longer than this will be truncated, sequences shorter will be padded.", )
parser.add_argument("--sc_per_gpu_train_batch_size", default=4, type=int,
                    help="Batch size per GPU/CPU for sequence classifier training.", )
parser.add_argument("--sc_per_gpu_eval_batch_size", default=4, type=int,
                    help="Batch size per GPU/CPU for sequence classifier evaluation.", )
parser.add_argument("--sc_per_gpu_unlabeled_batch_size", default=4, type=int,
                    help="Batch size per GPU/CPU for unlabeled examples used for distillation.", )
parser.add_argument("--sc_gradient_accumulation_steps", type=int, default=1,
                    help="Number of updates steps to accumulate before performing a backward/update pass for sequence "
                         "classifier training.", )
parser.add_argument("--sc_num_train_epochs", default=3, type=float,
                    help="Total number of training epochs to perform for sequence classifier training.", )
parser.add_argument("--sc_max_steps", default=-1, type=int,
                    help="If > 0: set total number of training steps to perform for sequence classifier training. "
                         "Override num_train_epochs.", )
parser.add_argument("--sc_min_steps", default=-1, type=int,
                    help="If > 0: set minimal number of steps. Use with num_train_epochs.")


# Other optional parameters
parser.add_argument("--train_examples", nargs="+", default=[-1], type=int)
parser.add_argument("--test_examples", default=-1, type=int,
                    help="The total number of test examples to use, where -1 equals all examples.", )
parser.add_argument("--unlabeled_examples", default=-1, type=int,
                    help="The total number of unlabeled examples to use, where -1 equals all examples", )
parser.add_argument("--split_examples_evenly", action="store_true",
                    help="If true, train examples are not chosen randomly, but split evenly across all labels.", )
parser.add_argument("--cache_dir", default="", type=str,
                    help="Where to store the pre-trained models downloaded from S3.")
parser.add_argument("--learning_rate", default=2e-5, type=float, help="The initial learning rate for Adam.")
parser.add_argument("--weight_decay", default=0.01, type=float, help="Weight decay if we apply some.")
parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
parser.add_argument("--logging_number", type=int, default=5, help="Log X times per training.")
parser.add_argument("--logging_steps", type=int, default=-1, help="Log every X updates steps. Overrides logging_number")
parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
parser.add_argument("--overwrite_output_dir", action="store_true", help="Overwrite the content of the output directory")
parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
parser.add_argument("--do_train", action="store_true", help="Whether to perform training")
parser.add_argument("--do_eval", action="store_true", help="Whether to perform evaluation")
parser.add_argument("--priming", action="store_true", help="Whether to use priming for evaluation")
parser.add_argument("--do_test", action="store_true",
                    help="Whether to perform evaluation on the test set", )

# fmt: on
