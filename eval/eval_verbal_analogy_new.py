import logging
import sys
import argparse
import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import re

# Automatically determine the correct path to the AbstractAnalogies directory
home_directory = os.path.expanduser('~')  # Gets the home directory
abstract_analogies_path = os.path.join(home_directory, 'AbstractAnalogies')
sys.path.append(abstract_analogies_path)

print(sys.path)

from models.llama3 import LLama3
from models.mistral7b import Mistral7B
from models.starling7b_beta import Starling7BBeta

SUPPORTED_MODELS = {
    'llama3': LLama3,
    'mistral7b': Mistral7B,
    'starling7b-beta': Starling7BBeta
}

def parse_option():
    parser = argparse.ArgumentParser("Evaluate Sentence Embedding Models")
    parser.add_argument(
        "--model", type=str, default="mistral7b", help="One of the models"
    )
    parser.add_argument(
        "--task", type=str, default="verbal_analogy"
    )
    parser.add_argument(
        "--prompt", type=str, default="basic_prompt.txt"
    )
    args = parser.parse_args()
    return args

def parse_model_generation(generation: str):
    # 1. Try finding <ans> </ans> tags
    pattern = r"<ans>(.*?)</ans>"

    # Find all matches
    matches = re.findall(pattern, generation)
    if len(matches) == 1 and (matches[0].strip() == 'D' or matches[0].strip() == 'E'):
        return matches[0].strip()
    
    # 2. Default to None if answer not found
    return None

def inference(model, rel, A, B, C, D, D_prime, prompt_template, results, task):
    prompt = prompt_template.format(A=A, B=B, C=C, D=D, D_prime=D_prime)
    generation = model.forward(prompt)
    parsed_answer = parse_model_generation(generation)

    ambiguous = True
    if parsed_answer == None:
        logit_D, logit_D_prime = model.forward_logits(prompt + ' So the final answer is <ans> ', task)
        parsed_answer = 'D' if logit_D > logit_D_prime else 'D_prime'
    else:
        ambiguous = False
        logit_D = None
        logit_D_prime = None

    results.append({
        'relation': rel,
        'A': A,
        'B': B,
        'C': C,
        'D': D,
        "D'": D_prime,
        'full_prompt': prompt,
        'raw_generation': generation,
        'parsed_answer': parsed_answer,
        'ambiguous': ambiguous,
        'logit_D': logit_D,
        "logit_D'": logit_D_prime
    })


def evaluate_verbal_analogies(args):
    print('Loading ', args.model)
    model_class = SUPPORTED_MODELS[args.model]
    # TODO: add passing config to model class init
    model = model_class()

    dataset = pd.read_csv('datasets/verbal_analogy/UCLA_VAT.csv')
    
    # read prompt_templates/story_analogies/basic_prompt.txt
    with open(f'prompt_templates/verbal_analogy/{args.prompt}', 'r', encoding='utf-8') as file:
        prompt_template = file.read()
    print(prompt_template)

    results = []
    for _, row in tqdm(dataset.iterrows(), total=len(dataset)):
        rel = row['Relation']
        A = row['A']
        B = row['B']
        C = row['C']
        D = row['D']
        D_prime = row["D'"]

        inference(model, rel, A, B, C, D, D_prime, prompt_template, results, args.task)

    # Check if results directory exists, if not, create it
    results_directory = './results'
    if not os.path.exists(results_directory):
        os.makedirs(results_directory)
    # Save results to csv
    prompt_format = args.prompt.split('.')[0]
    pd.DataFrame(results).to_csv(f'./results/verbal_analogies_logits_{args.model}_{prompt_format}.csv')


def main():
    args = parse_option()
    print(args)

    if args.task == 'verbal_analogy':
        evaluate_verbal_analogies(args)
    # TODO: implement other tasks


if __name__ == "__main__":
    main()
