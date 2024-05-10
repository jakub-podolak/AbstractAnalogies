import numpy as np
import pandas as pd
import builtins
import argparse
from prompt_templates.verbal_analogy import create_prompt
from models.llama3 import LLama3
from models.mistral7b import Mistral7B

SUPPORTED_MODELS = {
    'llama3': LLama3,
    'mistral7b': Mistral7B
}

def parse_option():
    parser = argparse.ArgumentParser("Evaluate Sentence Embedding Models")
    parser.add_argument(
        "--model", type=str, default="mistral7b", help="One of the models"
    )
    parser.add_argument(
        "--task", type=str, default="verbal_analogies"
    )
    args = parser.parse_args()
    return args

# pass mistral7b via args
args = parse_option()
model_class = SUPPORTED_MODELS[args.model]
model = model_class()

class VerbalAnalogyEvaluator:
    def __init__(self, data_path, results_path):
        """
        Initialize the VerbalAnalogyEvaluator class.
        Args:
			data_path (str): The path to the dataset.
			results_path (str): The path to save the results.
            
        Returns:
			None
        
        """
        df = pd.read_excel(data_path, sheet_name='UCLA_VAT')
        self.A = builtins.list(df['A'])
        self.B = builtins.list(df['B'])
        self.C = builtins.list(df['C'])
        self.D = builtins.list(df['D'])
        self.D_prime = builtins.list(df["D'"])

        self.results_path = results_path
        self.all_synonym_correct_pred = []
        self.all_opposite_correct_pred = []
        self.all_function_correct_pred = []
        self.all_category_correct_pred = []
        self.context = ""
        self.prob_order = np.arange(len(self.A))
        np.random.shuffle(self.prob_order)

    # def create_prompt(self, p, correct=True):
    #     """Create the prompt based on whether it's the first or subsequent problems."""
    #     prompt = self.context + '\n\n' if self.context and p != 0 else ""
    #     prompt += f"{self.A[self.prob_order[p]]} : {self.B[self.prob_order[p]]} :: {self.C[self.prob_order[p]]} : "
    #     return prompt + (self.D[self.prob_order[p]] if correct else self.D_prime[self.prob_order[p]])

    def get_model_response(self, prompt):
        """
        Get the model response to a prompt.
        Args:
			prompt (str): The prompt to send to the model.
            
		Returns:
			response (dict): The model's response to the prompt. I think its a dict of logprobs and text_offset
        """

        # TO DO: Do inference with the model and return the response
        # Check are these responses same as what authors got from gpt
        _, log_probs, text_offsets = model.forward_with_details(prompt)
        
        response = {
            'choices': [
                {
                    'logprobs': {
                        'token_logprobs': log_probs,
                        'text_offset': text_offsets
                    }
                }
            ]
        }
        return response

    def evaluate_problem(self, p):
        """
        Evaluate a specific analogy problem.
        Args:
			p (int): The index of the analogy problem to evaluate.
	
		Returns:
			None
        """
		# Correct Prompt (d_prompt): This prompt concatenates strings from columns A, B, C, and D. 
        # The format is like this: A : B :: C : D. This is the expected correct answer based on the analogy rule being tested.
        correct_prompt = create_prompt(p, correct=True)
        response = self.get_model_response(correct_prompt)
        d_avg_logprob = self.calculate_average_logprob(correct_prompt, response)
        
		# Incorrect Prompt (d_prime_prompt): This prompt uses the same format but replaces D with D' from the dataset. 
        # The format becomes: A : B :: C : D'. Here, D' is specifically provided in our dataset as a plausible but incorrect answer, making it the foil.
        incorrect_prompt = create_prompt(p, correct=False)
        response = self.get_model_response(incorrect_prompt)
        d_prime_avg_logprob = self.calculate_average_logprob(incorrect_prompt, response)
        
		# Compare the average log probabilities of the correct and incorrect prompts to determine the model's prediction.
        # A true value means the model preferred the correct answer, and false means the model was misled by the foil.
        correct_pred = d_avg_logprob > d_prime_avg_logprob
        if self.prob_order[p] < 20: # Synonym
            self.all_synonym_correct_pred.append(correct_pred)
        elif 20 <= self.prob_order[p] < 40: # Opposite
            self.all_opposite_correct_pred.append(correct_pred)
        elif 40 <= self.prob_order[p] < 60: # Function
            self.all_function_correct_pred.append(correct_pred)
        else:
            self.all_category_correct_pred.append(correct_pred) # Category

        self.context = correct_prompt if correct_pred else incorrect_prompt

    def calculate_average_logprob(self, prompt, response):
        """Calculate the average log probability from the model's response."""

        # TO DO: Check if this 'response' return is similar structure to gpt3 for consistency
        first_token_ind = np.where(np.array(response['choices'][0]['logprobs']['text_offset']) <= len(prompt))[0][-1]
        return np.mean(response['choices'][0]['logprobs']['token_logprobs'][first_token_ind:])

    def evaluate_all(self):
        """Evaluate all analogy problems and save results."""
        for p in range(len(self.A)):
            print(f"{p + 1} of {len(self.A)}...")
            self.evaluate_problem(p)
            np.savez(self.results_path,
                     synonym=self.all_synonym_correct_pred,
                     opposite=self.all_opposite_correct_pred,
                     function=self.all_function_correct_pred,
                     category=self.all_category_correct_pred,
                     context=self.context,
                     prob_order=self.prob_order,
                     allow_pickle=True)
            

def main():
    args = parse_option()
    print(args)

    if args.task == 'verbal_analogies':
        evaluator = VerbalAnalogyEvaluator('datasets/verbal_analogy/UCLA_VAT.xlsx', './UCLA_VAT_results.npz')
        evaluator.evaluate_all()

# Example usage to fix once the model is implemented:
if __name__ == "__main__":
    main()
