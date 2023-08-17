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
This file contains the logic for loading training and test data for all tasks.
"""

import csv
import json
import os
import sys
import random
import ast

from abc import ABC, abstractmethod
from collections import defaultdict, Counter
from typing import List, Dict, Callable

import log
from pet import task_helpers
from pet.utils import InputExample

logger = log.get_logger("root")


def _shuffle_and_restrict(examples: List[InputExample], num_examples: int, seed: int = 42) -> List[InputExample]:
    """
    Shuffle a list of examples and restrict it to a given maximum size.

    :param examples: the examples to shuffle and restrict
    :param num_examples: the maximum number of examples
    :param seed: the random seed for shuffling
    :return: the first ``num_examples`` elements of the shuffled list
    """
    if 0 < num_examples < len(examples):
        random.Random(seed).shuffle(examples)
        examples = examples[:num_examples]
    return examples


class LimitedExampleList:
    def __init__(self, labels: List[str], max_examples=-1):
        """
        Implementation of a list that stores only a limited amount of examples per label.

        :param labels: the set of all possible labels
        :param max_examples: the maximum number of examples per label. This can either be a fixed number,
               in which case `max_examples` examples are loaded for every label, or a list with the same size as
               `labels`, in which case at most `max_examples[i]` examples are loaded for label `labels[i]`.
        """
        self._labels = labels
        self._examples = []
        self._examples_per_label = defaultdict(int)

        if isinstance(max_examples, list):
            self._max_examples = dict(zip(self._labels, max_examples))
        else:
            self._max_examples = {label: max_examples for label in self._labels}

    def is_full(self):
        """Return `true` iff no more examples can be added to this list"""
        for label in self._labels:
            if self._examples_per_label[label] < self._max_examples[label] or self._max_examples[label] < 0:
                return False
        return True

    def add(self, example: InputExample) -> bool:
        """
        Add a new input example to this list.

        :param example: the example to add
        :returns: `true` iff the example was actually added to the list
        """
        label = example.label
        if self._examples_per_label[label] < self._max_examples[label] or self._max_examples[label] < 0:
            self._examples_per_label[label] += 1
            self._examples.append(example)
            return True
        return False

    def to_list(self):
        return self._examples


class DataProcessor(ABC):
    """
    Abstract class that provides methods for loading training, testing, development and unlabeled examples for a given
    task
    """

    @abstractmethod
    def get_train_examples(self, data_dir) -> List[InputExample]:
        """Get a collection of `InputExample`s for the train set."""
        pass

    @abstractmethod
    def get_dev_examples(self, data_dir) -> List[InputExample]:
        """Get a collection of `InputExample`s for the dev set."""
        pass

    @abstractmethod
    def get_test_examples(self, data_dir) -> List[InputExample]:
        """Get a collection of `InputExample`s for the test set."""
        pass

    @abstractmethod
    def get_unlabeled_examples(self, data_dir) -> List[InputExample]:
        """Get a collection of `InputExample`s for the unlabeled set."""
        pass

    @abstractmethod
    def get_labels(self) -> List[str]:
        """Get the list of labels for this data set."""
        pass

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf-8-sig") as f:
            # reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            reader = csv.reader(f, delimiter="\t")
            lines = []
            for line in reader:
                if sys.version_info[0] == 2:
                    line = list(cell.unicode('utf-8') for cell in line)
                lines.append(line)
            return lines


class MnliProcessor(DataProcessor):
    """Processor for the MultiNLI data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        return self._create_examples(MnliProcessor._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        return self._create_examples(MnliProcessor._read_tsv(os.path.join(data_dir, "dev_matched.tsv")), "dev_matched")

    def get_test_examples(self, data_dir) -> List[InputExample]:
        return self._create_examples(MnliProcessor._read_tsv(os.path.join(data_dir, "dev_mismatched.tsv")), "dev_mismatched")

    def get_unlabeled_examples(self, data_dir) -> List[InputExample]:
        return self.get_train_examples(data_dir)

    def get_labels(self):
        return ["contradiction", "entailment", "neutral"]

    @staticmethod
    def _create_examples(lines: List[List[str]], set_type: str) -> List[InputExample]:
        examples = []

        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            #change the index based on the dataset
            text_a = line[8]
            text_b = line[9]
            label = line[-1]

            example = InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label)
            examples.append(example)

        return examples


class CitationFunctionProcessor(DataProcessor):
    """Processor for the citation function classification set."""

    def get_train_examples(self, data_dir):
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.txt")), "train")

    def get_dev_examples(self, data_dir):
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "valid.txt")), "valid")

    def get_test_examples(self, data_dir) -> List[InputExample]:
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "test.txt")), "test")

    def get_unlabeled_examples(self, data_dir) -> List[InputExample]:
        #return self._create_examples_unlabeled(self._read_tsv(os.path.join(data_dir, "unlabeled.txt")), "unlabeled")
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.txt")), "train")

    def get_labels(self):
        return ["0", "1", "2", "3", "4", "5"]

    @staticmethod
    def _create_examples(lines: List[List[str]], set_type: str) -> List[InputExample]:
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue

            guid = "%s-%s" % (set_type, line[0])

            text_a = line[1]
            text_b = ""
            label = line[2]
            
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))

        return examples
        
        
    @staticmethod
    def _create_examples_unlabeled(lines: List[List[str]], set_type: str) -> List[InputExample]:
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue

            guid = "%s-%s" % (set_type, line[0])

            text_a = line[0]
            text_b = ""
            label = line[1]
            
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))

        return examples



PROCESSORS = {

    "function": CitationFunctionProcessor, 

}  # type: Dict[str,Callable[[],DataProcessor]]



METRICS = {"function": ["f1-macro", "f1-micro"], "function-generative": ["f1-macro", "f1-micro"]}

#DEFAULT_METRICS = ["f1-macro"]
DEFAULT_METRICS = ["acc"]

TRAIN_SET = "train"
DEV_SET = "dev"
TEST_SET = "test"
UNLABELED_SET = "unlabeled"

SET_TYPES = [TRAIN_SET, DEV_SET, TEST_SET, UNLABELED_SET]


def load_examples(
    task,
    data_dir: str,
    set_type: str,
    *_,
    num_examples: int = None,
    num_examples_per_label: int = None,
    seed: int = 42
) -> List[InputExample]:
    """Load examples for a given task."""
    assert (num_examples is not None) ^ (
        num_examples_per_label is not None
    ), "Exactly one of 'num_examples' and 'num_examples_per_label' must be set."
    assert (not set_type == UNLABELED_SET) or (
        num_examples is not None
    ), "For unlabeled data, 'num_examples_per_label' is not allowed"

    processor = PROCESSORS[task]()

    ex_str = (
        f"num_examples={num_examples}"
        if num_examples is not None
        else f"num_examples_per_label={num_examples_per_label}"
    )
    logger.info(f"Creating features from dataset file at {data_dir} ({ex_str}, set_type={set_type})")

    if set_type == DEV_SET:
        examples = processor.get_dev_examples(data_dir)
    elif set_type == TEST_SET:
        examples = processor.get_test_examples(data_dir)
    elif set_type == TRAIN_SET:
        examples = processor.get_train_examples(data_dir)
    elif set_type == UNLABELED_SET:
        examples = processor.get_unlabeled_examples(data_dir)
        for example in examples:
            example.label = processor.get_labels()[0]
    else:
        raise ValueError(f"'set_type' must be one of {SET_TYPES}, got '{set_type}' instead")

    if num_examples is not None:
        examples = _shuffle_and_restrict(examples, num_examples, seed)

    elif num_examples_per_label is not None:
        limited_examples = LimitedExampleList(processor.get_labels(), num_examples_per_label)
        for example in examples:
            limited_examples.add(example)
        examples = limited_examples.to_list()

    label_distribution = Counter(example.label for example in examples)
    logger.info(f"Returning {len(examples)} {set_type} examples with label dist.: {list(label_distribution.items())}")

    return examples
