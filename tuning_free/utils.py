import openai
from statistics import mean
import certifi
import json
import random
import numpy as np
import torch
from torch.utils.data import Dataset
import multiprocessing
import os
import sys
import csv
import pickle
import pandas as pd
#from nltk.tokenize import sent_tokenize
import ast
import time
import datetime
from openai.error import RateLimitError
import backoff


def convert_labels(label):

    if label == 0:
        return 'Background'
    elif label == 1:
        return 'Compares_Contrasts'
    elif label == 2:
        return 'Extension'
    elif label == 3:
        return 'Future'
    elif label == 4:
        return 'Motivation'
    elif label == 5:
        return 'Uses'
    else:
        raise ValueError('Label specified not correct')

def fix_seed(seed):
    # random
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # Pytorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def print_now(return_flag=0):
    t_delta = datetime.timedelta(hours=9)
    JST = datetime.timezone(t_delta, 'JST')
    now = datetime.datetime.now(JST)
    now = now.strftime('%Y/%m/%d %H:%M:%S')
    if return_flag == 0:
        print(now)
    elif return_flag == 1:
        return now
    else:
        pass

def read_tsv(input_file, quotechar=None):
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


def data_reader(args):
    if args.dataset == "act2":
        lines = read_tsv(args.input_file)
        unique_ids = []
        examples = []
        labels = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            
            uid = line[0]
            text_a = line[8]
            #text_a = " ".join(ast.literal_eval(line[9]))
            label = line[10]
            unique_ids.append(uid)
            examples.append(text_a)
            labels.append(label)
            
    if args.dataset == "acl_arc":
        lines = read_tsv(args.input_file)
        unique_ids = []
        examples = []
        labels = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            uid = line[0]
            text_a = line[7]
            label = line[9]
            unique_ids.append(uid)
            examples.append(text_a)
            labels.append(label)

    text_len_list = []
    for text in examples:
        text_len_list.append(len(text.split(" ")))

    q_len_mean = mean(text_len_list)
    print("dataset : {}".format(args.dataset))
    print("data size : {}".format(len(examples)))
    print("average num of words for each sample : {}".format(q_len_mean))

    return unique_ids, examples, labels

# Create dataset object before dataloader ...
class MyDataset(Dataset):
    
    def __init__(self, args):
        super().__init__()
        self.unique_ids, self.examples, self.labels = data_reader(args)
        self.len = len(self.examples)

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        unique_id = self.unique_ids[index]
        example = self.examples[index]
        label = self.labels[index]
        return unique_id, example, label
    

def setup_data_loader(args):
    # fix randomness of dataloader to ensure reproducibility
    # https://pytorch.org/docs/stable/notes/randomness.html
    fix_seed(args.random_seed)
    worker_seed = torch.initial_seed() % 2 ** 32
    print("worker_seed : {}".format(worker_seed))

    def seed_worker(worker_id):
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    g = torch.Generator()
    g.manual_seed(worker_seed)

    dataloader_num_workers = multiprocessing.cpu_count()
    dataloader_num_workers = min(dataloader_num_workers, args.max_num_worker)
    print("dataloader_num_workers: " + str(dataloader_num_workers))

    dataset = MyDataset(args)

    dataloader = torch.utils.data.DataLoader(dataset,
                                             shuffle=True,
                                             batch_size=args.minibatch_size,
                                             drop_last=False,
                                             num_workers=dataloader_num_workers,
                                             worker_init_fn=seed_worker,
                                             generator=g,
                                             pin_memory=True)
    """
    dataloader = torch.utils.data.DataLoader(dataset,
                                             shuffle=False,
                                             batch_size=args.minibatch_size,
                                             drop_last=False,
                                             num_workers=dataloader_num_workers,
                                             pin_memory=True)
    """

    return dataloader


class Decoder():
    def __init__(self, args):
        print_now()

    def decode(self, args, input, max_length, i, k):
        response = decoder_for_gpt3(args, input, max_length, i, k)
        return response


@backoff.on_exception(backoff.expo, RateLimitError)
def decoder_for_gpt3(args, input):

    openai.api_type = ""
    openai.api_version = " "
    openai.api_version = " "
    openai.api_base = " "
    openai.api_key = " "

    time.sleep(args.api_time_interval)
    if args.model == "gpt3.5":
        model = "gpt-35-turbo"
        engine = " " #add the engine

    else:
        model = "gpt-4"
        engine = " " #add the engine

    if args.method == "few_shot" or args.method == "zero_shot":
        messages = [
            {"role": "system", "content": "You are a helpful assistant that can provide information."},
            {"role": "user", "content": "Q: What is the function of the citation, '#CITATION_TAG' in the citation context: "
                                         f"{input}?"
                                          "Select the most appropriate option from the following:"
                                        "1) Background - #CITATION_TAG provides relevant background information or is part of the body of literature, 2) Uses - Citing paper uses the methodology or tools created by #CITATION_TAG, 3) Extension - Citing paper extends the methods, tools, or data in #CITATION_TAG, 4) Motivation - Citing paper is directly motivated by #CITATION_TAG, 5) Future - #CITATION_TAG is a potential avenue for future work, 6) Compares_Contrasts - Citing paper expresses similarities to or differences from, or disagrees with #CITATION_TAG"
                                          "A: Therefore, the answer is "
            },
        ]
        
    elif args.method == "few_shot_no_instruction" or args.method == "zero_shot_no_instruction":
        messages = [
            {"role": "system", "content": "You are a helpful assistant that can provide information."},
            {"role": "user", "content": "Q: What is the function of the citation, '#CITATION_TAG' in the citation context: "
                                         f"{input}?"
                                          "Select the most appropriate option from the following:"
                                        "1) Background - #CITATION_TAG provides relevant background information or is part of the body of literature, 2) Uses - Citing paper uses the methodology or tools created by #CITATION_TAG, 3) Extension - Citing paper extends the methods, tools, or data in #CITATION_TAG, 4) Motivation - Citing paper is directly motivated by #CITATION_TAG, 5) Future - #CITATION_TAG is a potential avenue for future work, 6) Compares_Contrasts - Citing paper expresses similarities to or differences from, or disagrees with #CITATION_TAG"
            },
        ]

    elif args.method == "few_shot_reason" or args.method == "zero_shot_reason":
        messages = [
            {"role": "system", "content": "You are a helpful assistant that can provide information."},
            {"role": "user", "content": "Q: Why is citation '#CITATION_TAG' cited in the sentence: "
                                         f"{input}?"
                                          "Select the most appropriate option from the following:"
                                        "1) Background - #CITATION_TAG provides relevant background information or is part of the body of literature, 2) Uses - Citing paper uses the methodology or tools created by #CITATION_TAG, 3) Extension - Citing paper extends the methods, tools, or data in #CITATION_TAG, 4) Motivation - Citing paper is directly motivated by #CITATION_TAG, 5) Future - #CITATION_TAG is a potential avenue for future work, 6) Compares_Contrasts - Citing paper expresses similarities to or differences from, or disagrees with #CITATION_TAG"
                                          "A: Therefore, the answer is "
            },
        ]

    elif args.method == "few_shot_description" or args.method == "zero_shot_description":
        messages = [
            {"role": "system", "content": "You are a helpful assistant that can provide information."},
            {"role": "user", "content": "Q: Given the citation context: "
                                        f"{input}"
                                        ", which of the following options is the most appropriate way to categorize #CITATION_TAG according to the author's reason for citing it?"
                                        "1) Background - #CITATION_TAG provides relevant background information or is part of the body of literature, 2) Uses - Citing paper uses the methodology or tools created by #CITATION_TAG, 3) Extension - Citing paper extends the methods, tools, or data in #CITATION_TAG, 4) Motivation - Citing paper is directly motivated by #CITATION_TAG, 5) Future - #CITATION_TAG is a potential avenue for future work, 6) Compares_Contrasts - Citing paper expresses similarities to or differences from, or disagrees with #CITATION_TAG"
                                          "A: Therefore, the answer is"
            },
        ]

    elif args.method == "zero_shot_hint_annotation":
        
        messages = [
            {"role": "system", "content": "You are a helpful assistant that can provide information."},
            {"role": "user", "content": "Imagine this scenario: You have been assigned the task of annotating citations within citation contexts to determine their citation functions. The citation function represents the author's motive for citing a specific paper. The citation context may contain one or more citations. Your goal is to annotate a specific citation, marked as '#CITATION_TAG', by assigning it to the most suitable class from the following options: "
                                        "1) Background - #CITATION_TAG provides relevant background information or is part of the body of literature, 2) Uses - Citing paper uses the methodology or tools created by #CITATION_TAG, 3) Extension - Citing paper extends the methods, tools, or data in #CITATION_TAG, 4) Motivation - Citing paper is directly motivated by #CITATION_TAG., 5) Future - #CITATION_TAG is a potential avenue for future work, 6) Compares_Contrasts - Citing paper expresses similarities to or differences from, or disagrees with #CITATION_TAG"
                                        "Considering the given citation context: "
                                        f"{input}"
                                        "Please select the most appropriate category to annotate #CITATION_TAG from the provided options."
                                        "A: Therefore, the answer is"},
        ]

    elif args.method == "few_shot_cot" or args.method == "zero_shot_cot":
        messages = [
            {"role": "system", "content": "You are a helpful assistant that can provide information."},
            {"role": "user", "content": "Q: What is the function of the citation, '#CITATION_TAG' in the citation context: "
                                         f"{input}?"
                                          "Select the most appropriate option from the following:"
                                        "A) Background - #CITATION_TAG provides relevant background information or is part of the body of literature, B) Uses - Citing paper uses the methodology or tools created by #CITATION_TAG, C) Extension - Citing paper extends the methods, tools, or data in #CITATION_TAG, D) Motivation - Citing paper is directly motivated by #CITATION_TAG, E) Future - #CITATION_TAG is a potential avenue for future work, F) Compares_Contrasts - Citing paper expresses similarities to or differences from, or disagrees with #CITATION_TAG"
                                          "Let's think step by step"
                                         
            },
        ]

    elif args.method == "zero_shot_hint":

        messages = [
            {"role": "system", "content": "Q: Why is citation '#CITATION_TAG' cited in the sentence: "
                                         f"{input}"
                                          "select answer from 'Background, Compares_Contrasts, Extension, Future, Motivation or Uses'"
                                          "hint: Background - '#CITATION_TAG provides relevant background information or is part of the body of literature', Uses - 'Citing paper uses the methodology or tools created by #CITATION_TAG', Extension - 'Citing paper extends the methods, tools, or data in #CITATION_TAG', Motivation - 'Citing paper is directly motivated by #CITATION_TAG', Future - '#CITATION_TAG is a potential avenue for future work.' Compares_Contrasts - 'Citing paper expresses similarities to or differences from, or disagrees with #CITATION_TAG.'"
                                          "A: Therefore, the answer is "
            },
            {"role": "user", "content": f"{input}"},
        ]
        
    elif args.method == "few_shot_cot_hint" or args.method == "zero_shot_cot_hint":
        
        messages = [
            {"role": "system", "content": "Q: Why is citation '#CITATION_TAG' cited in the sentence: "
                                         f"{input}"
                                          "select a single answer from 'Background, Compares_Contrasts, Extension, Future, Motivation or Uses'"
                                          "hint: Background - '#CITATION_TAG provides relevant background information or is part of the body of literature', Uses - 'The citing paper uses the methodology or tools created by #CITATION_TAG', Extension - 'The citing paper extends the methods, tools, or data in #CITATION_TAG', Motivation - 'The citing paper is directly motivated by #CITATION_TAG', Future - '#CITATION_TAG is a potential avenue for future work.' Compares_Contrasts - 'The citing paper expresses similarities to or differences from, or disagrees with #CITATION_TAG.'"
                                          "A: Let’s work this out in a step by step way to be sure we have the right answer."
            },
            {"role": "user", "content": f"{input}"},
        ]

    elif args.method == "zero_shot_cot_hint_new":
        
        messages = [
            {"role": "system", "content": "Q: Given the citation context: "
                                        f"{input}"
                                        ", which of the following options is the most appropriate way to categorize #CITATION_TAG according to the author's reason for citing it?"
                                        "1) Background - #CITATION_TAG provides relevant background information or is part of the body of literature, 2) Uses - Citing paper uses the methodology or tools created by #CITATION_TAG, 3) Extension - Citing paper extends the methods, tools, or data in #CITATION_TAG, 4) Motivation - Citing paper is directly motivated by #CITATION_TAG, 5) Future - #CITATION_TAG is a potential avenue for future work, 6) Compares_Contrasts - Citing paper expresses similarities to or differences from, or disagrees with #CITATION_TAG"
                                          "A: Therefore, the answer is"
            },
            {"role": "user", "content": f"{input}"},
        ]
        
    elif args.method == "zero_shot_hint_instruction":
        
        messages = [
            {"role": "system", "content": "Given the citation context: "
                                        f"{input}"
                                        ", choose the most appropriate way to categorize #CITATION_TAG based on the following options"
                                        "1) Background - #CITATION_TAG provides relevant background information or is part of the body of literature, 2) Uses - Citing paper uses the methodology or tools created by #CITATION_TAG, 3) Extension - Citing paper extends the methods, tools, or data in #CITATION_TAG, 4) Motivation - Citing paper is directly motivated by #CITATION_TAG, 5) Future - #CITATION_TAG is a potential avenue for future work, 6) Compares_Contrasts - Citing paper expresses similarities to or differences from, or disagrees with #CITATION_TAG"
                                        
            },
            {"role": "user", "content": f"{input}"},
        ]

    elif args.method == "zero_shot_hint_instruction_trigger":
        
        messages = [
            {"role": "system", "content": "Given the citation context: "
                                        f"{input}"
                                        ", choose the most appropriate way to categorize #CITATION_TAG based on the following options"
                                        "1) Background - #CITATION_TAG provides relevant background information or is part of the body of literature, 2) Uses - Citing paper uses the methodology or tools created by #CITATION_TAG, 3) Extension - Citing paper extends the methods, tools, or data in #CITATION_TAG, 4) Motivation - Citing paper is directly motivated by #CITATION_TAG, 5) Future - #CITATION_TAG is a potential avenue for future work, 6) Compares_Contrasts - Citing paper expresses similarities to or differences from, or disagrees with #CITATION_TAG"
                                        "Therefore, the answer is"
                                        
            },
            {"role": "user", "content": f"{input}"},
        ]
        
    elif args.method == "zero_shot_hint_instruction_cot":
        
        messages = [
            {"role": "system", "content": "Given the citation context: "
                                        f"{input}"
                                        ", choose the most appropriate way to categorize #CITATION_TAG based on the following options"
                                        "1) Background - #CITATION_TAG provides relevant background information or is part of the body of literature, 2) Uses - Citing paper uses the methodology or tools created by #CITATION_TAG, 3) Extension - Citing paper extends the methods, tools, or data in #CITATION_TAG, 4) Motivation - Citing paper is directly motivated by #CITATION_TAG, 5) Future - #CITATION_TAG is a potential avenue for future work, 6) Compares_Contrasts - Citing paper expresses similarities to or differences from, or disagrees with #CITATION_TAG"
                                        "A: Let’s work this out in a step by step way to be sure we have the right answer."
                                        
            },
            {"role": "user", "content": f"{input}"},
        ]

    else:
        print("Method not implemented")

    response = openai.ChatCompletion.create(
        engine=engine,
        messages=messages
    )
    
    return response["choices"][0]["message"]["content"].replace("\"", "").replace(":", "").replace(".", "")







    
    


