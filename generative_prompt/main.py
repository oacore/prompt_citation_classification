import argparse
import logging
import torch
import random
import time
import string
import requests
import os
from sklearn.metrics import f1_score
from utils import *


VERBALIZER = {"0": ["Background"], "1": ["Compares_Contrasts"], "2": ["Extension"], "3": ["Future"],
              "4": ["Motivation"], "5": ["Uses"]}


def verbalize(label, type):
    if type == 0:
        for l, p in VERBALIZER.items():
            if p[0] == label:
                return l

    else:
        for l, a in VERBALIZER.items():
            if l == str(label):
                return a[0]

def strip_punctuation(input_text):
    # Create translation table with punctuation characters mapped to None
    translator = str.maketrans("", "", string.punctuation)
    # Use translation table to remove punctuations from the input string
    stripped_string = input_text.translate(translator).lower()
    return stripped_string


# for longer outputs from chatgpt, check for citation function
def find_last_occurrence(output, classes):

    try:
        words = output.split()
        words_updated = strip_punctuation(output)
        words_updated = words_updated.split()
        
        for cls in classes:
            new_cls = ''.join(cls.lower().strip(string.punctuation).split('_'))
            if cls in words:
                return cls

            elif new_cls in words_updated:
                return cls

    except AttributeError as e:
        return -1

    return -1  # If the word is not found


def verify_predictions(pred, actual):
    classes = ['Background', 'Compares_Contrasts', 'Extension', 'Future', 'Motivation', 'Uses']
    label = None
    type = 0
    if pred not in classes:

        lbl = find_last_occurrence(pred, classes)
        if lbl != -1:
            label = verbalize(lbl, type)

        else:
            type = 1
            actual = verbalize(actual, type)
            while True:
                selected_value = random.choice(classes)
                if selected_value != actual:
                    type = 0
                    label = verbalize(selected_value, type)
                    break

    else:
        label = verbalize(pred, type)

    return label


def parse_arguments():
    parser = argparse.ArgumentParser(description="Zero-shot-CoT")

    parser.add_argument(
        "--api_log_file_name", type=str, default=None,
        help="mandatory argument ! json['i>=1']['j==1']['k={1,2}'][{'request', response'}]"
    )
    parser.add_argument(
        "--input_file", type=str, default=None, help="test data file path"
    )
    
    parser.add_argument(
        "--output_dir", type=str, default=None, help="test data file path"
    )
    
    parser.add_argument(
        "--PIK_kNN", type=str, default=None, help="path to knn embedding"
    )
    
    parser.add_argument(
        "--train_file", type=str, default=None, help="train file path with demonstrations"
    )

    #parser.add_argument("--random_seed", type=int, default=42, help="random seed")

    parser.add_argument(
        "--dataset", type=str, default="act2", choices=["act2", "acl_arc"], help="dataset used for experiment"
    )

    parser.add_argument("--minibatch_size", type=int, default=1, choices=[1],
                        help="minibatch size should be 1 because GPT-3 API takes only 1 input for each request")

    parser.add_argument("--max_num_worker", type=int, default=3, help="maximum number of workers for dataloader")

    parser.add_argument(
        "--model", type=str, default="gpt3", choices=["gpt3", "gpt3-medium", "gpt3-large", "gpt3-xl", "gpt3.5", "gpt4"],
        help="model used for decoding. Note that 'gpt3' are the smallest models."
    )

    parser.add_argument(
        "--method", type=str, default="generative_prompt_two",
        choices=["generative_prompt_one", "generative_prompt_two", "generative_prompt_four"], help="method"
    )

    parser.add_argument(
        "--max_length_cot", type=int, default=128,
        help="maximum length of output tokens by model for reasoning extraction"
    )
    parser.add_argument(
        "--max_length_direct", type=int, default=1,
        help="maximum length of output tokens by model for answer extraction"
    )
    parser.add_argument(
        "--limit_dataset_size", type=int, default=0,
        help="whether to limit test dataset size. if 0, the dataset size is unlimited and we use all the samples in the dataset for testing."
    )
    parser.add_argument(
        "--api_time_interval", type=float, default=1.0, help=""
    )
    parser.add_argument(
        "--log_dir", type=str, default="./log/", help="log directory"
    )

    args = parser.parse_args()
    #args.direct_answer_trigger = "\nTherefore, the answer is"
    # "Therefore, the answer ..." -> "The answer ..."
    #trigger = args.direct_answer_trigger.replace("\nTherefore, ", "")
    #args.direct_answer_trigger_for_zeroshot = trigger[0].upper() + trigger[1:]
    #args.direct_answer_trigger_for_zeroshot_cot = args.direct_answer_trigger

    return args

def main():
    args = parse_arguments()
    print('*****************************')
    print(args)
    print('*****************************')

    #fix_seed(args.random_seed)

    print("OPENAI_API_KEY:")
    print(os.getenv("OPENAI_API_KEY"))

    # Initialize decoder class (load model and tokenizer) ...

    decoder_gen = Decoder_gen(args)
    
    print("setup data loader ...")
    dataloader = setup_data_loader(args)
    print_now()
    
    total = 0
    correct_list = []
    to_write_predicted = []

    if not os.path.exists(os.path.join(args.output_dir, args.method)):
        os.makedirs(os.path.join(args.output_dir, args.method))
        print("Directory created!")
    else:
        print("Directory already exists!")
    output_file = open(os.path.join(args.output_dir, args.method, str(args.dataset)+"_"+str(args.model)+"_"+'output.csv'), 'w+')
    output_file_test = open(os.path.join(args.output_dir, args.method, str(args.dataset)+"_"+str(args.model)+"_"+'output_test.csv'), 'w+', newline='')
    
    output_file.write('unique_id,citation_context,actual_label'+'\n')
    output_file_test.write('unique_id,citation_context,actual_label'+'\n')
    predictions = csv.writer(output_file)
    predictions_test = csv.writer(output_file_test)
    try:
        for i, data in enumerate(dataloader):
            print('*************************')
            print("{}st data".format(i + 1))

            uid, citing_title, citing_abstract, cited_title, cited_abstract, x, y = data
            #X = "Q: " + x[0] + "\n" + "A:"
            uid = uid[0]
            X = x[0] 
            y = y[0].strip()
            title_citing = citing_title[0]
            title_cited = cited_title[0]
            
            abstract_citing = citing_abstract[0]
            abstract_cited = cited_abstract[0]
            
            max_length = args.max_length_direct
            z= decoder_gen.decode(args, title_citing, title_cited, X, abstract_citing, abstract_cited, max_length)
            pred = z

            updated_context = x[0] + " " + pred 
            to_write_predicted.append([uid,updated_context,y[0]])
            predictions_test.writerow([uid,updated_context,y[0]])

    except requests.exceptions.RequestException as e:
        print("A requests exception occurred:", str(e))
    
    except openai.OpenAIError as e:
        print("An OpenAI API exception occurred:", str(e))
    
    except Exception as e:
        print("An unexpected exception occurred:", str(e))

    predictions.writerows(to_write_predicted)
    output_file.close()
    output_file_test.close()


if __name__ == "__main__":
    main()