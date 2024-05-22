import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import builtins


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

def human_accuracy():
    # Human data
    human_file = os.path.join("datasets/verbal_analogies", "UCLA_VAT_ind_subj_data.xlsx")
    df = pd.read_excel (human_file, sheet_name='ind_subj')
    category_ind_subj_acc = np.array(builtins.list(df['category'])[1:]) / 100.
    function_ind_subj_acc = np.array(builtins.list(df['function'])[1:]) / 100.
    opposite_ind_subj_acc = np.array(builtins.list(df['opposite'])[1:]) / 100.
    synonym_ind_subj_acc = np.array(builtins.list(df['synonym'])[1:]) / 100.
    human_ind_subj = np.array([category_ind_subj_acc, function_ind_subj_acc, opposite_ind_subj_acc, synonym_ind_subj_acc])
    human_acc_across_categories = human_ind_subj.mean(1)
    total_human_acc = human_acc_across_categories.mean()

    # human_err = sem(human_ind_subj,1)
    return total_human_acc

human_accuracy_val = human_accuracy()
print(f"Human accuracy: {human_accuracy_val}")

# Define the order of models and prompts explicitly
models = ["mistral7b", "llama3"]
prompts = ["basic_prompt_not_forced", "basic_prompt_forced", "cot", "cot_structured"]

# Prepare accuracy data in the correct format
accuracy_data = {model: [] for model in models}
for model in models:
    for prompt in prompts:
        accuracy_data[model].append(accuracy_dict[(model, prompt)])

# Number of models and prompts
n_models = len(models)
n_prompts = len(prompts)

palette = 'pastel'

colors = {
    'basic_prompt_not_forced': sns.color_palette(palette)[0],
    'basic_prompt_forced': sns.color_palette(palette)[1],
    'cot': sns.color_palette(palette)[2],
    'cot_structured':sns.color_palette(palette)[3],
}


# Set width of bar to a smaller value for thinner bars
bar_width = 0.08

# Set positions of bar on X axis
r = np.arange(n_models)
positions = [r + i * bar_width for i in range(n_prompts)]


# Create figure and axes
fig, ax = plt.subplots(figsize=(12, 8))

# Plot each model's data
for i, prompt in enumerate(prompts):
    # positions_with_offset = positions[i] + offsets[prompt]
    positions_with_offset = positions[i]
    
    ax.bar(positions_with_offset, [accuracy_dict[(model, prompt)] for model in models], 
           width=bar_width, label=prompt, color=colors[prompt])


# Add labels
font_size = 20
font = {'family': 'monospace'}
ax.set_xlabel('Models', fontweight='bold', fontsize=font_size, **font)
ax.set_ylabel('Accuracy', fontweight='bold', fontsize=font_size, **font)
# ax.set_title('Verbal Analogies (Majority Vote from 3 prompt variants)')
ax.set_xticks(r + bar_width * (n_prompts - 1) / 2)
ax.set_xticklabels(models, fontsize=font_size, **font)
ax.set_yticks(np.arange(0, 1.1, 0.1))
# size of the ticks
ax.tick_params(axis='both', which='major', labelsize=12)
ax.axhline(y=human_accuracy_val, linewidth=2, label='Mean accuracy of human participants', linestyle='dashed')
# Legend here is about prompts
ax.legend(title='Prompt Types', loc='lower center', fontsize=12)

# Show plot
plt.show()

# Save the plot
plt.savefig(os.path.join(base_dir, "plot_accuracy_verbal_analogy.png"), dpi = 1200)

