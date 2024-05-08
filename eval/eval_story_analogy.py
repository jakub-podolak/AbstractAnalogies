import logging
import sys
import argparse
import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import re

import sys
sys.path.append('/home/scur0241/AbstractAnalogies')
from models.llama3 import LLama3
from models.mistral7b import Mistral7B

SUPPORTED_MODELS = {
    'llama3': LLama3,
    'mistral7b': Mistral7B
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


def evaluate_story_analogies(args):
    print('Loading ', args.model)
    model_class = SUPPORTED_MODELS[args.model]
    # TODO: add passing config to model class init
    model = model_class()

    dataset = pd.read_csv('datasets/story_analogies/story_analogies.csv')
    
    # read prompt_templates/story_analogies/basic_prompt.txt
    with open(f'prompt_templates/story_analogies/{args.prompt}', 'r', encoding='utf-8') as file:
        prompt_template = file.read()
    print(prompt_template)

    # TODO: add some shuffling and make sure it's correct according to the paper
    results = []
    for _, row in tqdm(dataset.iterrows(), total=len(dataset)):
        source_story = row['Base']
        correct_analogy = row['True Analogy Story']
        false_analogy = row['False Analogy Story']

        prompt = prompt_template.format(SourceStory=source_story, StoryA=correct_analogy, StoryB=false_analogy)

        generation = model.forward(prompt)
        parsed_answer = parse_model_generation(generation)

        results.append({
            'source_story': source_story,
            'correct_analogy': correct_analogy,
            'false_analogy': false_analogy,
            'full_prompt': prompt,
            'raw_generation': generation,
            'parsed_answer': parsed_answer,
            'correct_answer': 'A' # TODO: shuffle it randomly,
        })

    # Save results to csv
    pd.DataFrame(results).to_csv('story_analogies_result.csv')


def main():
    args = parse_option()
    print(args)

    if args.task == 'story_analogies':
        evaluate_story_analogies(args)
    # TODO: implement other tasks


if __name__ == "__main__":
    main()
