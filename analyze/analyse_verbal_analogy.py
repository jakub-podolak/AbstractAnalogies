import os

results_directory = './results'

verbal_results_directory = os.path.join(results_directory, 'verbal_analogies')
prompt_type = []
file_suffix =  ["prompt_templates_verbal_analogies" + prompt_type]


model_list = []


csv_file = os.path.join(verbal_results_directory, f'verbal_analogies_{each_model}_{file_suffix}.csv')
