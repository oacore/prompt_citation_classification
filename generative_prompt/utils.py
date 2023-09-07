import openai
from statistics import mean
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


def select_demos_knn(index_list, train_df):
    
    demonstrations = list()
    for i in index_list:
        selected_rows = train_df.iloc[i]
        label = convert_labels(selected_rows['pred_label'])
        data = {
                'citation_context': selected_rows['citation_context'],
                'rationale': selected_rows['predicted'],
                'predicted_label': label
            }
        demonstrations.append(data)
        
    return demonstrations


def process_messages(messages):
    total_length = sum(len(msg['content']) for msg in messages)
    if total_length <= 4096:
        return messages  # No processing necessary

    processed_messages = []
    accumulated_length = 0

    for msg in messages:
        content = msg['content']
        content_length = len(content)

        if accumulated_length + content_length <= 4096:
            processed_messages.append(msg)
            accumulated_length += content_length
        else:
            remaining_length = 4096 - accumulated_length
            truncated_content = content[:remaining_length]
            msg['content'] = truncated_content + '...'
            processed_messages.append(msg)
            break

    return processed_messages


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
        citing_abstracts = []
        cited_abstracts = []
        citing_titles = []
        cited_titles = []
        examples = []
        labels = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            
            uid = line[0]
            citing_title = line[2]
            citing_abstract = line[4]
            cited_title = line[5]
            cited_abstract = line[7] 
            text_a = line[8]
            #text_a = " ".join(ast.literal_eval(line[9]))
            label = line[10]
            unique_ids.append(uid)
            citing_titles.append(citing_title)
            citing_abstracts.append(citing_abstract)
            cited_titles.append(cited_title)
            cited_abstracts.append(cited_abstract)
            examples.append(text_a)
            labels.append(label)
            
    if args.dataset == "acl_arc":
        lines = read_tsv(args.input_file)
        unique_ids = []
        citing_abstracts = []
        cited_abstracts = []
        citing_titles = []
        cited_titles = []
        examples = []
        labels = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            uid = line[0]
            citing_title = line[2]
            citing_abstract = line[3]
            cited_title = line[4]
            cited_abstract = line[6] 
            text_a = line[7]
            label = line[9]
            unique_ids.append(uid)
            citing_titles.append(citing_title)
            citing_abstracts.append(citing_abstract)
            cited_titles.append(cited_title)
            cited_abstracts.append(cited_abstract)
            examples.append(text_a)
            labels.append(label)

    text_len_list = []
    for text in examples:
        text_len_list.append(len(text.split(" ")))

    q_len_mean = mean(text_len_list)
    print("dataset : {}".format(args.dataset))
    print("data size : {}".format(len(examples)))
    print("average num of words for each sample : {}".format(q_len_mean))

    return unique_ids, citing_titles, citing_abstracts, cited_titles, cited_abstracts, examples, labels

# Create dataset object before dataloader ...
class MyDataset(Dataset):
    
    def __init__(self, args):
        super().__init__()
        self.unique_ids, self.citing_titles , self.citing_abstracts, self.cited_titles , self.cited_abstracts, self.examples, self.labels = data_reader(args)
        self.len = len(self.examples)

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        unique_id = self.unique_ids[index]
        citing_title = self.citing_titles[index]
        citing_abstract = self.citing_abstracts[index]
        cited_title = self.cited_titles[index]
        cited_abstract = self.cited_abstracts[index]
        example = self.examples[index]
        label = self.labels[index]
        return unique_id, citing_title, citing_abstract, cited_title, cited_abstract, example, label
    

def setup_data_loader(args):

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

    return dataloader


class Decoder_gen():
    def __init__(self, args):
        print_now()

    def decode(self, args, citing_title, cited_title, citation_context, citing_abstract, cited_abstract, max_length):
        response = decoder_for_gpt3(args, citing_title, cited_title, citation_context, citing_abstract, cited_abstract, max_length)
        return response


@backoff.on_exception(backoff.expo, RateLimitError)
def decoder_for_gpt3(args, citing_title, cited_title, citation_context, citing_abstract, cited_abstract, max_length):
    openai.api_type = ""
    openai.api_version = " "
    openai.api_version = " "
    openai.api_base = " "
    openai.api_key = " "

    time.sleep(args.api_time_interval)
    if args.model == "gpt3.5":
        model = "gpt-35-turbo"
        engine = " "  # add the engine

    else:
        model = "gpt-4"
        engine = " "  # add the engine
    if args.method == "generative_prompt_four":
        
        messages = [
            {"role": "system", "content": "Given the following information, "
                                        "Citing sentence containing citation, represented by #CITATION_TAG: "
                                        f"{citation_context},"
                                        "Citing paper title: "
                                        f"{citing_title}, "
                                        "Cited paper title: "
                                        f"{cited_title}"
                                        "Citing paper abstract:"
                                        f"{citing_abstract}"
                                        "Cited paper abstract:"
                                        f"{cited_abstract}"
                                        "please note that cited paper abstract is not always available"
                                       
                                        
            },
            {"role": "user", "content": "explain the relationship between citing paper and cited paper in a single sentence?"},
        ]

        #messages = messages[:4096]
        messages = process_messages(messages)

    elif args.method == "generative_prompt_two":
        
        messages = [
            {"role": "system", "content": "Given the following information, "
                                        "Citing sentence from citing paper contains citation to the cited paper, represented by #CITATION_TAG: "
                                        f"{citation_context},"
                                        "Citing paper title: "
                                        f"{citing_title}, "
                                        "Cited paper title: "
                                        f"{cited_title}"
            
            },
            {"role": "user", "content": "explain the relationship between citing paper and cited paper in a single sentence?"},
        ]

    elif args.method == "generative_prompt_one":
        
        messages = [
            {"role": "system", "content": "Given the following information, "
                                        "Citing sentence from citing paper contains citation to the cited paper, represented by #CITATION_TAG: "
                                        f"{citation_context},"
                                        "Cited paper title: "
                                        f"{cited_title}"
                                    
            },
            {"role": "user", "content": "explain the relationship between citing paper and cited paper in a single sentence?"},
        ]
    
    else:
        print("Method not implemented")
    
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages
    )

    return response["choices"][0]["message"]["content"].replace("\"", "").replace(":", "").replace(".", "")

    


