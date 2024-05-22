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

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(f"Root directory: {root_dir}")
sys.path.insert(0, root_dir)
print(f"sys.path: {sys.path}")

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
        "--model", type=str, default="starling7b-beta", help="One of the models"
    )
    parser.add_argument(
        "--task", type=str, default="verbal_analogy"
    )
    parser.add_argument(
        "--prompt", type=str, default="prompt_templates/verbal_analogies/3_cot/1.txt"
    )
    args = parser.parse_args()
    return args

def parse_model_generation(generation: str):
    # 1. Try finding <ans> </ans> tags
    pattern = r"<ans>\s*([A-Z])[^<]*</ans>"

    # Find all matches
    matches = re.findall(pattern, generation)
    if len(matches) == 1 and (matches[0].strip() == 'D' or matches[0].strip() == 'E'):
        return matches[0].strip()
    
    # 2. Default to None if answer not found
    return None

def parse_extended_model_generation(extended_prompt, generation: str):
    extended_prompt = extended_prompt
    # 1. Try finding <ans> </ans> tags
    # pattern = r"<ans>(.*?)</ans>"
    pattern = r"<ans>\s*([A-Z])\s*</ans>"

    # Find all matches
    matches = re.findall(pattern, generation)
    if len(matches) == 1 and (matches[0].strip() == 'D' or matches[0].strip() == 'E'):
        return matches[0].strip()
    # Extract the first character from generation
    # if len(generation) > 0:
    #     return generation[0]
    else:
        return None

def inference(model, rel, A, B, C, D, D_prime, prompt_template, results, task):
    prompt = prompt_template.format(A=A, B=B, C=C, D=D, D_prime=D_prime)
    generation = model.forward(prompt)
    parsed_answer = parse_model_generation(generation)

    ambiguous = False
    if parsed_answer == None:
        ambiguous = True
        # extended_prompt = prompt + "\n" + generation + "\n So the final answer as a single letter is <ans> "
        extended_prompt = prompt + "\n" + generation + "\n So the final answer in single letter is (return just <ans> D </ans> or <ans> E </ans>): "

        new_generation = model.forward(extended_prompt)
        parsed_answer = parse_extended_model_generation(extended_prompt, new_generation)

    results.append({
        'relation': rel,
        'A': A,
        'B': B,
        'C': C,
        'D': D,
        "D'": D_prime,
        'raw_generation': generation,
        'new_generation': new_generation if ambiguous else None,
        'parsed_answer': parsed_answer,
    })


def evaluate_verbal_analogies(args):
    print('Loading ', args.model)
    model_class = SUPPORTED_MODELS[args.model]
    # TODO: add passing config to model class init
    model = model_class()

    dataset = pd.read_csv('datasets/verbal_analogies/UCLA_VAT.csv')
    
    # read prompt_templates/story_analogies/basic_prompt.txt
    with open(f'{args.prompt}', 'r', encoding='utf-8') as file:
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


    verbal_results_directory = os.path.join(results_directory, 'verbal_analogies')
    if not os.path.exists(verbal_results_directory):
        os.makedirs(verbal_results_directory)

    # Save results to csv
    prompt_format = args.prompt.split('.')[0]
    # create ddf of resutts and save results to csv
    fname_from_prompt = prompt_format.split("/")
    file_prefix = '_'.join(fname_from_prompt)
    csv_file = os.path.join(verbal_results_directory, f'verbal_analogies_{args.model}_{file_prefix}.csv')
    # Delete the file if it already exists
    if os.path.exists(csv_file):
        os.remove(csv_file)
    df = pd.DataFrame(results)
    df.to_csv(csv_file, index=False)


def main():
    args = parse_option()
    print(args)

    if args.task == 'verbal_analogy':
        evaluate_verbal_analogies(args)
    # TODO: implement other tasks


if __name__ == "__main__":
    main()
