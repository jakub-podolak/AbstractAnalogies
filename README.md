# AbstractAnalogies

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