# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
This file contains code for wrapping a transformer language model and
provides convenience methods for training and inference.
"""
import json
import os
from math import floor, ceil
from typing import List, Dict, Optional

import jsonpickle
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, accuracy_score
from torch.utils.data import RandomSampler, DataLoader, DistributedSampler
from tqdm import trange, tqdm
from transformers import (
    InputExample,
    AdamW,
    get_linear_schedule_with_warmup,
    PreTrainedTokenizer,
    BertForMaskedLM,
    RobertaForMaskedLM,
    BertConfig,
    BertForSequenceClassification,
    BertTokenizer,
    GPT2Config,
    GPT2LMHeadModel,
    GPT2Tokenizer,
    AutoConfig,
    AutoTokenizer,
    AutoModelForMaskedLM,
    AutoModelForSequenceClassification, 
    AutoModel
)
from transformers import __version__ as transformers_version
from transformers.data.metrics import simple_accuracy

import log
from pet import preprocessor
from pet.tasks import TASK_HELPERS
from pet.utils import InputFeatures, DictDataset, distillation_loss, distributed_concat, exact_match
#from pet.modeling import *
#from pet.modeling import strict_accuracy, weak_accuracy

logger = log.get_logger("root")

CONFIG_NAME = "wrapper_config.json"
SEQUENCE_CLASSIFIER_WRAPPER = "sequence_classifier"
MLM_WRAPPER = "mlm"
PLM_WRAPPER = "plm"

WRAPPER_TYPES = [SEQUENCE_CLASSIFIER_WRAPPER, MLM_WRAPPER, PLM_WRAPPER]


PREPROCESSORS = {
    SEQUENCE_CLASSIFIER_WRAPPER: preprocessor.SequenceClassifierPreprocessor,
    MLM_WRAPPER: preprocessor.MLMPreprocessor,
    PLM_WRAPPER: preprocessor.PLMPreprocessor,
}


MODEL_CLASSES = {
    "bert": {
        "config": BertConfig,
        "tokenizer": BertTokenizer,
        SEQUENCE_CLASSIFIER_WRAPPER: BertForSequenceClassification,
        MLM_WRAPPER: BertForMaskedLM,
    },
    "scibert": {
        "config": AutoConfig,
        "tokenizer": AutoTokenizer,
        SEQUENCE_CLASSIFIER_WRAPPER: AutoModelForSequenceClassification,
        MLM_WRAPPER: AutoModelForMaskedLM,
    },
    "specter": {
        "config": AutoConfig,
        "tokenizer": AutoTokenizer,
        SEQUENCE_CLASSIFIER_WRAPPER: AutoModelForSequenceClassification,
        MLM_WRAPPER: AutoModelForMaskedLM,
    },

    "gpt2": {"config": GPT2Config, "tokenizer": GPT2Tokenizer, MLM_WRAPPER: GPT2LMHeadModel},
}

EVALUATION_STEP_FUNCTIONS = {
    MLM_WRAPPER: lambda wrapper: wrapper.mlm_eval_step,
    PLM_WRAPPER: lambda wrapper: wrapper.plm_eval_step,
    SEQUENCE_CLASSIFIER_WRAPPER: lambda wrapper: wrapper.sequence_classifier_eval_step,
}

TRAIN_STEP_FUNCTIONS = {
    MLM_WRAPPER: lambda wrapper: wrapper.mlm_train_step,
    PLM_WRAPPER: lambda wrapper: wrapper.plm_train_step,
    SEQUENCE_CLASSIFIER_WRAPPER: lambda wrapper: wrapper.sequence_classifier_train_step,
}


class WrapperConfig(object):
    """A configuration for a :class:`TransformerModelWrapper`."""

    def __init__(
            self,
            model_type: str,
            model_name_or_path: str,
            wrapper_type: str,
            task_name: str,
            max_seq_length: int,
            label_list: List[str],
            pattern_id: int = 0,
            verbalizer_file: str = None,
            cache_dir: str = None,
    ):
        """
        Create a new config.

        :param model_type: the model type (e.g., 'bert', 'roberta', 'albert')
        :param model_name_or_path: the model name (e.g., 'roberta-large') or path to a pretrained model
        :param wrapper_type: the wrapper type (one of 'mlm', 'plm' and 'sequence_classifier')
        :param task_name: the task to solve
        :param max_seq_length: the maximum number of tokens in a sequence
        :param label_list: the list of labels for the task
        :param pattern_id: the id of the pattern to use
        :param verbalizer_file: optional path to a verbalizer file
        :param cache_dir: optional path to a cache dir
        """
        self.model_type = model_type
        self.model_name_or_path = model_name_or_path
        self.wrapper_type = wrapper_type
        self.task_name = task_name
        self.max_seq_length = max_seq_length
        self.label_list = label_list
        self.pattern_id = pattern_id
        self.verbalizer_file = verbalizer_file
        self.cache_dir = cache_dir
        self.loggers_initialized = False


class TransformerModelWrapper:
    """A wrapper around a Transformer-based language model."""

    def __init__(self, config: WrapperConfig):
        """Create a new wrapper from the given config."""
        self.config = config
        config_class = MODEL_CLASSES[self.config.model_type]["config"]
        tokenizer_class = MODEL_CLASSES[self.config.model_type]["tokenizer"]
        model_class = MODEL_CLASSES[self.config.model_type][self.config.wrapper_type]

        model_config = config_class.from_pretrained(
            config.model_name_or_path,
            num_labels=len(config.label_list),
            finetuning_task=config.task_name,
            cache_dir=config.cache_dir if config.cache_dir else None,
            use_cache=False,
        )

        self.tokenizer = tokenizer_class.from_pretrained(
            config.model_name_or_path, cache_dir=config.cache_dir if config.cache_dir else None
        )  # type: PreTrainedTokenizer

        if self.config.model_type == "gpt2":
            self.tokenizer.pad_token, self.tokenizer.mask_token = self.tokenizer.eos_token, self.tokenizer.eos_token

        self.model = model_class.from_pretrained(
            config.model_name_or_path, config=model_config, cache_dir=config.cache_dir if config.cache_dir else None
        )

        self.preprocessor = PREPROCESSORS[self.config.wrapper_type](
            self, self.config.task_name, self.config.pattern_id, self.config.verbalizer_file
        )
        self.task_helper = TASK_HELPERS[self.config.task_name](self) if self.config.task_name in TASK_HELPERS else None

    @classmethod
    def from_pretrained(cls, path: str) -> "TransformerModelWrapper":
        """Load a pretrained wrapper from a given path."""
        wrapper = TransformerModelWrapper.__new__(TransformerModelWrapper)
        wrapper.config = wrapper._load_config(path)
        tokenizer_class = MODEL_CLASSES[wrapper.config.model_type]["tokenizer"]
        model_class = MODEL_CLASSES[wrapper.config.model_type][wrapper.config.wrapper_type]
        wrapper.model = model_class.from_pretrained(path)
        wrapper.tokenizer = tokenizer_class.from_pretrained(path)
        wrapper.preprocessor = PREPROCESSORS[wrapper.config.wrapper_type](
            wrapper, wrapper.config.task_name, wrapper.config.pattern_id, wrapper.config.verbalizer_file
        )
        wrapper.task_helper = (
            TASK_HELPERS[wrapper.config.task_name](wrapper) if wrapper.config.task_name in TASK_HELPERS else None
        )
        return wrapper

    def save(self, path: str) -> None:
        """Save a pretrained wrapper."""
        model_to_save = self.model.module if hasattr(self.model, "module") else self.model
        model_to_save.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        self._save_config(path)

    def _save_config(self, path: str) -> None:
        with open(os.path.join(path, CONFIG_NAME), "w") as f:
            f.write(jsonpickle.encode(self.config))

    @staticmethod
    def _load_config(path: str) -> WrapperConfig:
        with open(os.path.join(path, CONFIG_NAME), "r") as f:
            return jsonpickle.decode(f.read())

    def train(
            self,
            task_train_data: List[InputExample],
            device,
            per_gpu_train_batch_size: int = 8,
            n_gpu: int = 1,
            num_train_epochs: int = 3,
            gradient_accumulation_steps: int = 1,
            weight_decay: float = 0.0,
            learning_rate: float = 5e-5,
            adam_epsilon: float = 1e-8,
            warmup_steps=0,
            max_grad_norm: float = 1,
            logging_steps: int = -1,
            logging_number: int = 5,
            per_gpu_unlabeled_batch_size: int = 8,
            unlabeled_data: List[InputExample] = None,
            lm_training: bool = False,
            use_logits: bool = False,
            alpha: float = 0.8,
            temperature: float = 1,
            max_steps=-1,
            min_steps=-1,
            output_dir=None,
            eval_kwargs=None,
            local_rank=-1,
            **_
    ):
        """
        Train the underlying language model.

        :param task_train_data: the training examples to use
        :param device: the training device (cpu/gpu)
        :param per_gpu_train_batch_size: the number of training examples per batch and gpu
        :param n_gpu: the number of gpus to use
        :param num_train_epochs: the number of epochs to train
        :param gradient_accumulation_steps: the number of gradient accumulation steps before performing an update
        :param weight_decay: the weight decay to use
        :param learning_rate: the learning rate to use
        :param adam_epsilon: epsilon parameter for the Adam optimizer
        :param warmup_steps: the number of warmup steps
        :param max_grad_norm: the maximum norm for the gradient
        :param logging_steps: the number of steps after which logging information is printed
        :param per_gpu_unlabeled_batch_size: the number of unlabeled examples per batch and gpu
        :param unlabeled_data: the unlabeled examples to use
        :param lm_training: whether to perform auxiliary language modeling (only for MLMs)
        :param use_logits: whether to use the example's logits instead of their labels to compute the loss
        :param alpha: the alpha parameter for auxiliary language modeling
        :param temperature: the temperature for knowledge distillation
        :param max_steps: the maximum number of training steps, to use with ``num_train_epochs``
        :param min_steps: the minimum number of training steps, to use with ``num_train_epochs``
        :return: a tuple consisting of the total number of steps and the average training loss
        """

        train_batch_size = per_gpu_train_batch_size * max(1, n_gpu)
        train_dataset = self._generate_dataset(task_train_data)
        if local_rank != -1:
            train_sampler = DistributedSampler(train_dataset)
        else:
            train_sampler = RandomSampler(train_dataset)
        train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=train_batch_size)
        #train_dataloader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle = False)

        unlabeled_dataloader, unlabeled_iter = None, None

        if lm_training or use_logits:
            # we need unlabeled data both for auxiliary language modeling and for knowledge distillation
            assert unlabeled_data is not None
            unlabeled_batch_size = per_gpu_unlabeled_batch_size * max(1, n_gpu)
            unlabeled_dataset = self._generate_dataset(unlabeled_data, labelled=False)
            unlabeled_sampler = RandomSampler(unlabeled_dataset)
            unlabeled_dataloader = DataLoader(
                unlabeled_dataset, sampler=unlabeled_sampler, batch_size=unlabeled_batch_size
            )
            #unlabeled_dataloader = DataLoader(
           #     unlabeled_dataset, batch_size=unlabeled_batch_size, shuffle = False
            #)
           
            unlabeled_iter = unlabeled_dataloader.__iter__()

        if use_logits:
            train_dataloader = unlabeled_dataloader

        if min_steps > 0:
            num_train_epochs = max(
                num_train_epochs, floor(min_steps / max(1, len(train_dataloader) / gradient_accumulation_steps))
            )

        if max_steps > 0:
            num_train_epochs = min(
                num_train_epochs, floor(max_steps / max(1, len(train_dataloader) / gradient_accumulation_steps))
            )

        if num_train_epochs == 0:
            num_train_epochs = 1

        t_total = len(train_dataloader) // gradient_accumulation_steps * num_train_epochs

        if logging_steps < 0:
            logging_steps = t_total // logging_number

        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]

        optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total
        )

        # multi-gpu training
        if local_rank != -1:
            self.model = torch.nn.parallel.DistributedDataParallel(self.model, device_ids=[local_rank])
        if n_gpu > 1:
            self.model = torch.nn.DataParallel(self.model)

        # The number of passes
        batch_step = 0
        # The number of gradient updates
        accumulated_step = 0
        tr_loss, logging_loss = 0.0, 0.0
        best_score = -1
        self.model.zero_grad()

        train_iterator = trange(int(num_train_epochs), desc="Epoch")

        for _ in train_iterator:
            epoch_iterator = tqdm(train_dataloader, desc="Iteration")
            for step, batch in enumerate(epoch_iterator):
                batch_step += 1
                self.model.train()
                unlabeled_batch = None

                batch = {k: t.to(device) for k, t in batch.items()}

                if lm_training:
                    while unlabeled_batch is None:
                        try:
                            unlabeled_batch = unlabeled_iter.__next__()
                        except StopIteration:
                            logger.info("Resetting unlabeled dataset")
                            unlabeled_iter = unlabeled_dataloader.__iter__()

                    lm_input_ids = unlabeled_batch["input_ids"]
                    unlabeled_batch["input_ids"], unlabeled_batch["mlm_labels"] = self._mask_tokens(lm_input_ids)
                    unlabeled_batch = {k: t.to(device) for k, t in unlabeled_batch.items()}

            
                train_step_inputs = {
                    "unlabeled_batch": unlabeled_batch,
                    "lm_training": lm_training,
                    "alpha": alpha,
                    "use_logits": use_logits,
                    "temperature": temperature,
                }
                
                loss = self.task_helper.train_step(batch, **train_step_inputs) if self.task_helper else None

                if loss is None:
                    loss = TRAIN_STEP_FUNCTIONS[self.config.wrapper_type](self)(batch, **train_step_inputs)
                    #loss = loss_function(batch, **train_step_inputs)
                if n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu parallel training
                if gradient_accumulation_steps > 1:
                    loss = loss / gradient_accumulation_steps

                loss.backward()

                tr_loss += loss.item()
                if batch_step % gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
                    optimizer.step()
                    scheduler.step()
                    self.model.zero_grad()
                    accumulated_step += 1

                    if logging_steps > 0 and accumulated_step % logging_steps == 0:
                        logs = {}
                        loss_scalar = (tr_loss - logging_loss) / logging_steps
                        learning_rate_scalar = scheduler.get_lr()[0]
                        logs["learning_rate"] = learning_rate_scalar
                        logs["loss"] = loss_scalar
                        logging_loss = tr_loss

                        if eval_kwargs is not None:
                            score = self.in_training_eval(eval_kwargs)
                            if score > best_score:
                                logger.info(f"New best score {score} > {best_score} previously, saving model")
                                self.save(output_dir)
                                best_score = score
                            else:
                                logger.info(f"Old best score {best_score} >= {score} currently, not saving model")

                        print(json.dumps({**logs, **{"step": accumulated_step}}))

        if eval_kwargs is not None:
            score = self.in_training_eval(eval_kwargs)
            if score > best_score:
                logger.info(f"New best score {score} > {best_score} previously, saving model")
                self.save(output_dir)
            else:
                logger.info(f"Old best score {best_score} >= {score} currently, not saving model")

        return accumulated_step, (tr_loss / accumulated_step if accumulated_step > 0 else -1)

    def eval(
            self,
            eval_data: List[InputExample],
            device,
            per_gpu_eval_batch_size: int = 8,
            n_gpu: int = 1,
            priming: bool = False,
            decoding_strategy: str = "default",
            local_rank=-1,
            **_
    ) -> Dict:
        """
        Evaluate the underlying language model.

        :param eval_data: the evaluation examples to use
        :param device: the evaluation device (cpu/gpu)
        :param per_gpu_eval_batch_size: the number of evaluation examples per batch and gpu
        :param n_gpu: the number of gpus to use
        :param priming: whether to use priming
        :param decoding_strategy: the decoding strategy for PET with multiple masks ('default', 'ltr' or 'parallel')
        :return: a dictionary of numpy arrays containing the indices, logits, labels, and (optional) question_ids for
                 each evaluation example.
        """

        eval_dataset = self._generate_dataset(eval_data, priming=priming)
        eval_batch_size = per_gpu_eval_batch_size * max(1, n_gpu)
        if local_rank != -1:
            eval_sampler = DistributedSampler(eval_dataset, shuffle=False)
        else:
            eval_sampler = RandomSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=eval_batch_size)
        #eval_dataloader = DataLoader(eval_dataset, batch_size=eval_batch_size, shuffle=False)

        if n_gpu > 1:
            self.model = torch.nn.DataParallel(self.model)
        if local_rank != -1:
            self.model = torch.nn.parallel.DistributedDataParallel(self.model, device_ids=[local_rank])

        preds = None
        all_indices, out_label_ids, question_ids = None, None, None

        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            self.model.eval()

            batch = {k: t.to(device) for k, t in batch.items()}
            labels = batch["labels"]
            indices = batch["idx"]
            with torch.no_grad():

                # some tasks require special evaluation
                logits = (
                    self.task_helper.eval_step(batch, decoding_strategy=decoding_strategy)
                    if self.task_helper
                    else None
                )

                if logits is None:
                    logits = EVALUATION_STEP_FUNCTIONS[self.config.wrapper_type](self)(batch)

            if preds is None:
                preds = logits
                out_label_ids = labels
                all_indices = indices
                if "question_idx" in batch:
                    question_ids = batch["question_idx"]
            else:
                preds = torch.cat((preds, logits), dim=0)
                out_label_ids = torch.cat((out_label_ids, labels), dim=0)
                all_indices = torch.cat((all_indices, indices), dim=0)
                if "question_idx" in batch:
                    question_ids = torch.cat((question_ids, batch["question_idx"]), dim=0)

        if local_rank != -1:
            preds = (
                distributed_concat(preds, num_total_examples=len(eval_data), interleave=True).detach().cpu().numpy()
            )
            out_label_ids = (
                distributed_concat(out_label_ids, num_total_examples=len(eval_data), interleave=True)
                    .detach()
                    .cpu()
                    .numpy()
            )
            all_indices = (
                distributed_concat(all_indices, num_total_examples=len(eval_data), interleave=True)
                    .detach()
                    .cpu()
                    .numpy()
            )
            if "question_idx" in batch:
                question_ids = (
                    distributed_concat(question_ids, num_total_examples=len(eval_data), interleave=True)
                        .detach()
                        .cpu()
                        .numpy()
                )
        else:
            preds = preds.detach().cpu().numpy()
            out_label_ids = out_label_ids.detach().cpu().numpy()
            all_indices = all_indices.detach().cpu().numpy()
            if "question_idx" in batch:
                question_ids = question_ids.detach().cpu().numpy()

        return {"indices": all_indices, "logits": preds, "labels": out_label_ids, "question_ids": question_ids}

    def in_training_eval(self, eval_kwargs):
        eval_results = self.eval(**eval_kwargs)
        predictions = np.argmax(eval_results["logits"], axis=1)

        if eval_kwargs["metrics"]:
            if "f1" in eval_kwargs["metrics"]:
                score = f1_score(eval_results["labels"], predictions)
            elif "f1-macro" in eval_kwargs["metrics"]:
                score = f1_score(eval_results["labels"], predictions, average="macro")
            elif "em" in eval_kwargs["metrics"]:
                score = exact_match(
                    predictions, eval_results["labels"], eval_results["question_ids"]
                )

            else:
                score = simple_accuracy(predictions, eval_results["labels"])
        else:
            score = simple_accuracy(predictions, eval_results["labels"])

        return score
    
    
    
    def _generate_dataset(self, data: List[InputExample], labelled: bool = True, priming: bool = False):
        features = self._convert_examples_to_features(data, labelled=labelled, priming=priming)
        feature_dict = {
            "input_ids": torch.tensor([f.input_ids for f in features], dtype=torch.long),
            "attention_mask": torch.tensor([f.attention_mask for f in features], dtype=torch.long),
            "token_type_ids": torch.tensor([f.token_type_ids for f in features], dtype=torch.long),
            "labels": torch.tensor([f.label for f in features], dtype=torch.long),
            "mlm_labels": torch.tensor([f.mlm_labels for f in features], dtype=torch.long),
            "logits": torch.tensor([f.logits for f in features], dtype=torch.float),
            "idx": torch.tensor([f.idx for f in features], dtype=torch.long),
        }
        if self.config.wrapper_type == PLM_WRAPPER:
            feature_dict["perm_mask"] = torch.tensor([f.perm_mask for f in features], dtype=torch.float)
            feature_dict["target_mapping"] = torch.tensor([f.target_mapping for f in features], dtype=torch.float)

        if self.task_helper:
            self.task_helper.add_features_to_dict(features, feature_dict)

        return DictDataset(**feature_dict)

    def _convert_examples_to_features(
            self, examples: List[InputExample], labelled: bool = True, priming: bool = False
    ) -> List[InputFeatures]:
        features = []
        for (ex_index, example) in enumerate(examples):
            if ex_index % 100000 == 0:
                logger.info("Writing example {}".format(ex_index))
            input_features = self.preprocessor.get_input_features(example, labelled=labelled, priming=priming)
            if self.task_helper:
                self.task_helper.add_special_input_features(example, input_features)
            features.append(input_features)
            # if ex_index < 5:
            #     logger.info(f'--- Example {ex_index} ---')
            #     logger.info(input_features.pretty_print(self.tokenizer))
        return features

    def _mask_tokens(self, input_ids):
        """ Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original. """
        labels = input_ids.clone()
        # We sample a few tokens in each sequence for masked-LM training (with probability 0.15)
        probability_matrix = torch.full(labels.shape, 0.15)
        special_tokens_mask = [
            self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
        ]
        probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)

        masked_indices = torch.bernoulli(probability_matrix).bool()

        # if a version of transformers < 2.4.0 is used, -1 is the expected value for indices to ignore
        if [int(v) for v in transformers_version.split(".")][:3] >= [2, 4, 0]:
            ignore_value = -100
        else:
            ignore_value = -1

        labels[~masked_indices] = ignore_value  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        input_ids[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        input_ids[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return input_ids, labels

    def generate_default_inputs(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Generate the default inputs required by almost every language model."""
        inputs = {"input_ids": batch["input_ids"], "attention_mask": batch["attention_mask"]}
        if self.config.model_type in ["bert", "scibert", "xlnet", "specter"]:
            inputs["token_type_ids"] = batch["token_type_ids"]
        return inputs

    def mlm_train_step(
            self,
            labeled_batch: Dict[str, torch.Tensor],
            unlabeled_batch: Optional[Dict[str, torch.Tensor]] = None,
            lm_training: bool = False,
            alpha: float = 0,
            **_
    ) -> torch.Tensor:
        """Perform a MLM training step."""

        inputs = self.generate_default_inputs(labeled_batch)
        mlm_labels, labels = labeled_batch["mlm_labels"], labeled_batch["labels"]

        outputs = self.model(**inputs)
        prediction_scores = self.preprocessor.pvp.convert_mlm_logits_to_cls_logits(mlm_labels, outputs[0])
        loss = nn.CrossEntropyLoss()(prediction_scores.view(-1, len(self.config.label_list)), labels.view(-1))

        if lm_training:
            lm_inputs = self.generate_default_inputs(unlabeled_batch)
            lm_inputs["masked_lm_labels"] = unlabeled_batch["mlm_labels"]
            lm_loss = self.model(**lm_inputs)[0]
            loss = alpha * loss + (1 - alpha) * lm_loss
        return loss

    def plm_train_step(self, labeled_batch: Dict[str, torch.Tensor], lm_training: bool = False, **_):
        """Perform a PLM training step."""

        inputs = self.generate_default_inputs(labeled_batch)
        inputs["perm_mask"], inputs["target_mapping"] = labeled_batch["perm_mask"], labeled_batch["target_mapping"]
        labels = labeled_batch["labels"]
        outputs = self.model(**inputs)
        prediction_scores = self.preprocessor.pvp.convert_plm_logits_to_cls_logits(outputs[0])
        loss = nn.CrossEntropyLoss()(prediction_scores.view(-1, len(self.config.label_list)), labels.view(-1))

        if lm_training:
            raise NotImplementedError("Language model training is currently not implemented for PLMs")

        return loss

    def sequence_classifier_train_step(
            self, batch: Dict[str, torch.Tensor], use_logits: bool = False, temperature: float = 1, **_
    ) -> torch.Tensor:
        """Perform a sequence classifier training step."""
        #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        #weight=torch.Tensor([0.30435841, 1.34843581, 2.91375291, 7.57575758, 1.78062678, 1.06837607]).to(device)
        """
        if args.data_type == "act2":
            weight=torch.FloatTensor([0.30435841, 1.34843581, 2.91375291, 7.57575758, 1.78062678, 1.06837607]).to(device)
        else:
            weight=torch.FloatTensor([0.32256169, 0.92424242, 4.65254237, 4.81578947, 3.8125, 0.88263666]).to(device)
        """
        inputs = self.generate_default_inputs(batch)
        if not use_logits:
            inputs["labels"] = batch["labels"]

        outputs = self.model(**inputs)
        """
        logits = outputs['logits']
        criterion = nn.CrossEntropyLoss(weight=weight)
        loss = criterion(logits, inputs['labels'])
        
        """
        if use_logits:
            logits_predicted, logits_target = outputs[0], batch["logits"]
            return distillation_loss(logits_predicted, logits_target, temperature)
        else:
            return outputs[0]
        """
        if use_logits:
            logits_predicted, logits_target = outputs[0], batch["logits"]
            return distillation_loss(logits_predicted, logits_target, temperature)
        else:
            return loss
        """
    def mlm_eval_step(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Perform a MLM evaluation step."""
        inputs = self.generate_default_inputs(batch)
        outputs = self.model(**inputs)
        return self.preprocessor.pvp.convert_mlm_logits_to_cls_logits(batch["mlm_labels"], outputs[0])

    def plm_eval_step(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Perform a PLM evaluation step."""
        inputs = self.generate_default_inputs(batch)
        inputs["perm_mask"], inputs["target_mapping"] = batch["perm_mask"], batch["target_mapping"]
        outputs = self.model(**inputs)
        return self.preprocessor.pvp.convert_plm_logits_to_cls_logits(outputs[0])

    def sequence_classifier_eval_step(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Perform a sequence classifier evaluation step."""
        inputs = self.generate_default_inputs(batch)
        return self.model(**inputs)[0]
        







    
    

