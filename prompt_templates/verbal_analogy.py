def create_prompt(self, p, correct=True):
    """
    Create the prompt based on whether it's the first or subsequent problems.
    Args:
        p (int): The index of the analogy problem to evaluate.
        correct (bool): Whether the prompt is for the correct answer or not.

    Returns:
        prompt (str): The prompt for the analogy problem.
    
    """
    prompt = self.context + '\n\n' if self.context and p != 0 else ""
    prompt += f"{self.A[self.prob_order[p]]} : {self.B[self.prob_order[p]]} :: {self.C[self.prob_order[p]]} : "
    return prompt + (self.D[self.prob_order[p]] if correct else self.D_prime[self.prob_order[p]])