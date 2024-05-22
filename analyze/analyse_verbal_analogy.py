import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


base_dir = "analyze"
combined_results_file = os.path.join(base_dir, "all_verbal_analogies.csv")

# if files doesnt exist, break and print error
if not os.path.exists(combined_results_file):
    print("Error: File does not exist")
    exit()

# read the combined file
df_combined = pd.read_csv(combined_results_file)

# for all values in the columns except ["relation", "A", "B", "C", "D"], if value is empty, replace with "E"
df_combined.fillna("E", inplace=True)

# get the columns except ["relation", "A", "B", "C", "D"]
columns = df_combined.columns[5:]

model_name = ["mistral7b", "llama3"]

# if first charatcter is a digit, remove it from of all the columns e.g. remove "1", "2" and "3" 
# also remove any leading or trailing underscores

new_columns = [column[1:] if column[0].isdigit() else column for column in columns]
new_columns = [column.strip("_") for column in new_columns]

# rename the columns
df_combined.columns = df_combined.columns[:5].tolist() + new_columns

# print(f"columns: {df_combined.columns}")

# for all columns which have the same model name in cilumn and [ "basic_prompt_not_forced", "basic_prompt_forced", "cot", "cot_structured"]
# get the majority vote of the answers in those 3 columns
for model in model_name:
    new_col_names = []
    for prompt in ["basic_prompt_not_forced", "basic_prompt_forced", "cot", "cot_structured"]:
        model_columns = [column for column in new_columns if model in column and prompt in column]

        # make new column with the majority vote
        # column name will be model + prompt
        df_combined[model + "_" + prompt] = df_combined[model_columns].mode(axis=1)[0]
        new_col_names.append(model + "_" + prompt)

# save the new dataframe to a new csv file and keep column: ["relation", "A", "B", "C", "D"] and the new columns
new_csv_file = os.path.join(base_dir, "majority_vote_verbal_analogies.csv")

if os.path.exists(new_csv_file):
    os.remove(new_csv_file)
    print("File already exists. Deleting file...")
    
df_combined.to_csv(new_csv_file, columns=["relation", "A", "B", "C", "D"] + new_col_names, index=False)

# Calculate the accuracy of the majority vote per model and prompt
# where D=correct answer, E=incorrect answer

accuracy_dict = {}
for model in model_name:
    for prompt in ["basic_prompt_not_forced", "basic_prompt_forced", "cot", "cot_structured"]:
        # get the column name
        col_name = model + "_" + prompt

        # get the accuracy
        accuracy = (df_combined[col_name] == "D").sum() / len(df_combined)
        accuracy_dict[(model, prompt)] = accuracy
        
        print(f"Model: {model}, Prompt: {prompt}, Accuracy: {accuracy}")


# Define the order of models explicitly
models = ["mistral7b", "llama3"]
prompts = sorted(set(prompt for _, prompt in accuracy_dict.keys()))

# Prepare accuracy data in the correct format
accuracy_data = {model: [] for model in models}
for prompt in prompts:
    for model in models:
        accuracy_data[model].append(accuracy_dict[(model, prompt)])

# Number of models and prompts
n_models = len(models)
n_prompts = len(prompts)

# Set width of bar
bar_width = 0.15

# Set positions of bar on X axis
r = np.arange(n_prompts)
positions = [r - bar_width/2, r + bar_width/2]

# Create figure and axes
fig, ax = plt.subplots(figsize=(10, 6))

# Plot each model's data
for i, model in enumerate(models):
    ax.bar(positions[i], accuracy_data[model], width=bar_width, edgecolor='grey', label=model)

# Add labels
ax.set_xlabel('Prompts', fontweight='bold')
ax.set_ylabel('Accuracy', fontweight='bold')
ax.set_title('Comparison of Accuracy across Different Prompts and Models')
ax.set_xticks(r)
ax.set_xticklabels(prompts)
# Legend should be about the models
ax.legend(models)

# Show plot
plt.show()


plt.savefig(os.path.join(base_dir, "accuracy_comparison_verbal_analogy.png"))