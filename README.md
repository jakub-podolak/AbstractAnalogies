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

## Source of Data:
- Story Analogies: can be downloaded from http://cvl.psych.ucla.edu/resources/AnalogyInventory.zip (file name: Cognitive Psychology.xlsx, sheet name: Rattermann). Based on https://github.com/taylorwwebb/emergent_analogies_LLM/blob/main/story_analogies/README.md.