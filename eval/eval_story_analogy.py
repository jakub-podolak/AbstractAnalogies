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
        "--task", type=str, default="story_analogies"
    )
    parser.add_argument(
        "--condition", type=str, default="far", help="Condition setting to use: near or far"
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
    if len(matches) == 1 and (matches[0].strip() == 'A' or matches[0].strip() == 'B'):
        return matches[0].strip()
    
    # 2. Default to None if answer not found
    return None

def inference(model, source_story, correct_analogy, false_analogy, prompt_template, results, task, correct_answer="A"):

    StoryA = correct_analogy if correct_answer == "A" else false_analogy
    StoryB = false_analogy if correct_answer == "A" else correct_analogy

    prompt = prompt_template.format(SourceStory=source_story, StoryA=StoryA, StoryB=StoryB)
    generation = model.forward(prompt)
    parsed_answer = parse_model_generation(generation)

    ambiguous = False
    if parsed_answer is None:
        ambiguous = True
        # Second-stage extraction
        extended_prompt = prompt + "\n" + generation + "\n So the final answer is (return just <ans> A </ans> or <ans> B </ans>):"
        new_generation = model.forward(extended_prompt)
        parsed_answer = parse_model_generation(new_generation)

    # ambiguous = True
    # if parsed_answer == None:
    #     logit_A, logit_B = model.forward_logits(prompt + ' So the final answer is <ans> ', task)
    #     parsed_answer = 'A' if logit_A > logit_B else 'B'
    # else:
    #     ambiguous = False
    #     logit_A = None
    #     logit_B = None

    results.append({
        'source_story': source_story,
        'story_A': StoryA,
        'story_B': StoryB,
        'full_prompt': prompt,
        'raw_generation': generation,
        'new_generation': new_generation if ambiguous else None,
        'parsed_answer': parsed_answer,
        'correct_answer': correct_answer,
        #'ambiguous': ambiguous,
        #'logit_A': logit_A,
        #'logit_B': logit_B
    })


def evaluate_story_analogies(args):
    print('Loading ', args.model)
    if '/' in args.prompt:
        prompt_format = args.prompt.replace('/', '-').split('.')[0]
    else:
        prompt_format = args.prompt.split('.')[0]
    print(prompt_format)

    model_class = SUPPORTED_MODELS[args.model]
    # TODO: add passing config to model class init
    model = model_class()

    dataset = pd.read_csv('datasets/story_analogies/story_analogies.csv')
    
    # read prompt_templates/story_analogies/basic_prompt.txt
    path = f'prompt_templates/story_analogies/{args.prompt}' if 'prompt_templates/' not in args.prompt else args.prompt
    with open(path, 'r', encoding='utf-8') as file:
        prompt_template = file.read()
    print(prompt_template)

    results = []
    for _, row in tqdm(dataset.iterrows(), total=len(dataset)):
        source_story = row['Base']
        if args.condition == "far":
            correct_analogy = row['True Analogy Story']
            false_analogy = row['False Analogy Story']
        elif args.condition == "near":
            correct_analogy = row['Literally similar story']
            false_analogy = row['Mere-Appearance Match']

        # inference as correct answer A
        inference(model, source_story, correct_analogy, false_analogy, prompt_template, results, args.task, correct_answer= "A")
        # inference as correct answer B
        inference(model, source_story, correct_analogy, false_analogy, prompt_template, results, args.task, correct_answer= "B")

    # Check if results directory exists, if not, create it
    results_directory = './results'
    if not os.path.exists(results_directory):
        os.makedirs(results_directory)
    # Save results to csv

    pd.DataFrame(results).to_csv(f'./results/story_far_all_prompts/story_analogies_{args.condition}_{args.model}_{prompt_format}.csv')


def main():
    args = parse_option()
    print(args)

    if args.task == 'story_analogies':
        evaluate_story_analogies(args)
    # TODO: implement other tasks


if __name__ == "__main__":
    main()
