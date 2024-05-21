#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --job-name=EvalStoryAnalogies_Starling7b
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --time=04:00:00
#SBATCH --output=eval_starling7b_story_analogies_two_stage_%A.out

module purge
module load 2022
module load Anaconda3/2022.05

source activate abstract

# Define the list of models
models=("starling7b-beta")  # Add your models to this list

# Define the directory to search for text files
directory="prompt_templates/story_analogies"  # Change this to your target directory

# Iterate over each model
for model in "${models[@]}"; do 
    echo "Starting model:"
    echo $model

    # Find all text files in the directory and its subdirectories
    find "$directory" -type f -name '*.txt' | while read -r file; do
        # Run the command for each text file with the current model
        echo $file
        srun python -u eval/eval_story_analogy.py --model "$model" --prompt "$file"
    done
done
