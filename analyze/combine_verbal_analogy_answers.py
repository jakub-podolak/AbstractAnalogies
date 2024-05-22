"""
This file is used to combine all the verbal analogy results into a single csv file.
Each column in the csv file corresponds to a model and the values are the parsed answers.
"""

import os
import pandas as pd
from tqdm import tqdm

results_directory = './results'
verbal_results_directory = os.path.join(results_directory, 'verbal_analogies')

# get all .csv files in the directory
csv_files = [f for f in os.listdir(verbal_results_directory) if f.endswith('.csv')]

# print the number of files
print('Number of files:', len(csv_files))
# print ('Files:', csv_files)


# Substrings to remove
substrings_to_remove = ["verbal_analogies", "prompt_templates"]

# Function to remove specified substrings from file names
def clean_file_suffix(file_names, substrings_to_remove):
    cleaned_file_suffix = []
    for name in file_names:
        model_name = name.split('.')[0].split('_')[2:3][0]

        for substring in substrings_to_remove:
            name = name.split(".")[0].replace(substring, "")

        # Remove model name
        name = name.replace(model_name, "") if model_name in name else name
        # Remove any leading or trailing underscores that might remain and prepend model name
        name =  name.strip("_")
        cleaned_file_suffix.append(name)
    return cleaned_file_suffix

# Function to get clean file suffix
file_suffix_list = []
file_suffix_list = clean_file_suffix(csv_files, substrings_to_remove)
# remove duplicates
file_suffix_list = list(dict.fromkeys(file_suffix_list))


# read the first file
df = pd.read_csv(os.path.join(verbal_results_directory, csv_files[0]))
df = df[["relation", "A", "B", "C", "D", "parsed_answer"]]

# check if any value in list file_suffix_list is in file name
column_suffix = next((s for s in file_suffix_list if s in csv_files[0]), None)
model_name = csv_files[0].split('.')[0].split('_')[2:3][0]
df.columns = ["relation", "A", "B", "C", "D", column_suffix + "_" + model_name]

# read the rest of the files
for file in tqdm(csv_files[1:]):
    model_name = file.split('.')[0].split('_')[2:3][0]
    # check if any value in list file_suffix_list is in file name
    column_suffix = next((s for s in file_suffix_list if s in file), None)
    column_name = column_suffix + "_" + model_name

    temp_df = pd.read_csv(os.path.join(verbal_results_directory, file))
    temp_df = temp_df[["parsed_answer"]]
    temp_df.columns = [column_name]
    df = pd.concat([df, temp_df], axis=1)

# save the new df
df.to_csv(os.path.join(results_directory, 'all_verbal_analogies.csv'), index=False)