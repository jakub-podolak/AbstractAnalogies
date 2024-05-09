import pandas as pd
from scipy.stats import binomtest

def calculate_accuracy(results_file):
    # Load the results from CSV
    results_df = pd.read_csv(results_file)
    
    # Calculate the number of correct predictions
    correct_predictions = (results_df['parsed_answer'] == results_df['correct_answer']).sum()
    
    # Calculate the total number of predictions
    total_predictions = len(results_df)
    
    # Calculate accuracy
    accuracy = correct_predictions / total_predictions
    
    # Perform binomial test and get confidence interval
    binom_result = binomtest(correct_predictions, total_predictions)
    confidence_interval = binom_result.proportion_ci(confidence_level=0.95)
    
    return accuracy, confidence_interval

def save_results_to_file(file_path, results):
    with open(file_path, 'w') as file:
        file.write("Model,Accuracy,95% Confidence Interval\n")
        for result in results:
            model_name = result['model'].split('/')[-1]
            # Format the confidence interval to avoid CSV format breaking
            ci_lower, ci_upper = result['confidence_interval']
            ci_formatted = f"[{ci_lower:.2f} - {ci_upper:.2f}]"
            file.write(f"{model_name},{result['accuracy']:.2f},{ci_formatted}\n")

results = []
models = [
    'results/story_analogies_far_logits_mistral7b_basic_prompt.csv',
    'results/story_analogies_far_logits_mistral7b_cot.csv',
    'results/story_analogies_far_logits_mistral7b_cot_structured.csv',
    'results/story_analogies_far_logits_llama3_basic_prompt.csv',
    'results/story_analogies_far_logits_llama3_cot.csv',
    'results/story_analogies_far_logits_llama3_cot_structured.csv'
]

for model in models:
    accuracy, confidence_interval = calculate_accuracy(model)
    results.append({
        'model': model,
        'accuracy': accuracy,
        'confidence_interval': confidence_interval
    })

save_results_to_file('analyze/accuracy_results.csv', results)

