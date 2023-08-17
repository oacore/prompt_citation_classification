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

import math
from abc import ABC
from collections import defaultdict
from typing import Dict, List, Optional, Any
import torch
import re

import numpy as np
from torch.nn import CrossEntropyLoss

from pet.utils import (
    InputFeatures,
    InputExample,
    get_verbalization_ids,
    chunks,
    trim_input_ids,
    remove_final_punc,
    lowercase_first,
)


class TaskHelper(ABC):
    """
    A helper class that provides custom training and evaluation methods for tasks that do not fit in PETs default
    schema, for example because they require more than two sequences of text, different evaluation metrics or
    verbalizers consisting of multiple tokens.
    """

    def __init__(self, wrapper):
        """
        Create a new task helper.

        :param wrapper: The wrapper for the language model being used.
        """
        self.wrapper = wrapper
        self.output = None

    def train_step(self, batch: Dict[str, torch.Tensor], **kwargs) -> Optional[torch.Tensor]:
        """
        Custom implementation of the train step for this task.

        :param batch: a batch of examples
        :return: a scalar loss tensor
        """
        pass

    def eval_step(self, batch: Dict[str, torch.Tensor], **kwargs) -> Optional[torch.Tensor]:
        """
        Custom implementation of the eval step for this task.

        :param batch: a batch of examples
        :return: a tensor of logits
        """
        pass

    def add_special_input_features(self, input_example: InputExample, input_features: InputFeatures) -> None:
        """
        Add special features to the ``meta`` dictionary of a feature set

        :param input_example: the input example considered
        :param input_features: the set of features corresponding to this example
        """

        pass

    def add_features_to_dict(self, features: List[InputFeatures], feature_dict: Dict[str, torch.Tensor]) -> None:
        """
        Add special features from the ``meta`` dictionary of a sequence of features to the corresponding dictionary

        :param features: the sequence of features
        :param feature_dict: the dictionary that stores aggregated feature views as tensors
        """
        pass

    def get_sequence_classifier_inputs(self, example: InputExample) -> Dict[str, Any]:
        """
        Get the inputs for sequence classification. Override this method if the input for the task considered is of a
        more complicated form than `text_a` or `text_a [SEP] text_b`.

        :param example: the input example
        :return: the dictionary of inputs
        """
        pass





