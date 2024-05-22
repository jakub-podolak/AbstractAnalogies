# AbstractAnalogies

This repository contains code for evaluating language models on story and verbal analogies. The code is based on the code from the paper "Emergent Analogies: A Method for Discovering the Structure of Analogical Relationships" by Webb et al. (2021). The code is adapted to work with the Hugging Face Transformers library and evaluate models on story and verbal analogies to test if "Chain of Reasoning" performs better than "Basic Prompt" for analogy completions.

Directory structure;
```
|-- analyze                  # code for analyzing results
|-- datasets                 # datasets for story and verbal analogies
   |-- story_analogies
   |-- verbal_analogies
|-- eval                     # code for evaluating models
|-- models                   # models to evaluate
   |-- mistral7b
   |-- llama3
   |-- starling7b
|-- prompt_templates         # prompts for story and verbal analogies
   |-- story_analogies
   |-- verbal_analogies
|-- results                  # results of evaluations
|-- abstract.yml             # conda environment file
|-- install_environment.job  # job file to install environment on Snellius
|-- README.md                # this file
```


## 1. Setup on Snellius
Install environment using
```
sbatch install_environment.job
```
This may take ~30 minutes

Run interactive session
```
srun --partition=gpu --gpus=1 --ntasks=1 --cpus-per-task=18 --time=04:00:00 --pty bash -i
```

And later
```
module purge
module load 2022
module load Anaconda3/2022.05

source activate abstract
```

You may be asked to login to huggingface before you start using some models like mistral7b or llama3. Then run
```
huggingface-cli login
```
And enter your token generated here: https://huggingface.co/settings/tokens

## Evaluate models / prompts

Run for example:
```
python3 eval/eval_story_analogy.py --model mistral7b --prompt basic_prompt.txt
```
Where prompt is the file with your prompt in directory `prompt_templates/{task}/`

## Source of Data:
- Story Analogies: can be downloaded from http://cvl.psych.ucla.edu/resources/AnalogyInventory.zip (file name: Cognitive Psychology.xlsx, sheet name: Rattermann). Based on https://github.com/taylorwwebb/emergent_analogies_LLM/blob/main/story_analogies/README.md.

- Verbal Analogies: can be downloaded from: https://github.com/taylorwwebb/emergent_analogies_LLM/tree/main/UCLA_VAT (file name: UCLA_VAT.xlsx). 