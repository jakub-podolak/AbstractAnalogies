#!/bin/bash

# Define the list of models
# models=("mistral7b" "starling7b-beta" "llama3" )  # Add your models to this list
models=("starling7b-beta")  # Add your models to this list


# Define the directory to search for text files
directory="prompt_templates/verbal_analogies"  # Change this to your target directory


# Iterate over each model
for model in "${models[@]}"; do 
    echo "Starting model:"
    echo $model
    # echo **** lines
    echo "****************************************************************"


    # Find all text files in the directory and its subdirectories
    find "$directory" -type f -name '*.txt' | while read -r file; do
        # Run the command for each text file with the current model
        echo $file
        python3 eval/eval_verbal_analogy.py --model "$model" --prompt "$file"
    done
done