from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

from abc import ABC

from models.main import EasyInferenceModel


class Mistral7B(EasyInferenceModel):
    def __init__(self, max_new_tokens=2048):
        self.device = "cuda" # the device to load the model onto

        model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

        self.model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2", torch_dtype=torch.float16, device_map="auto")

        self.tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")

        self.max_new_tokens = max_new_tokens

        print('Loaded mistral7b with device', self.model.device)

    
    def forward(self, text: str):
        messages = []
        messages.append({'role': 'user', 'content': text})

        encodeds = self.tokenizer.apply_chat_template(messages, return_tensors="pt")

        model_inputs = encodeds.to(self.model.device)
        generated_ids = self.model.generate(model_inputs, max_new_tokens=self.max_new_tokens, do_sample=False, pad_token_id=self.tokenizer.eos_token_id)
        decoded = self.tokenizer.batch_decode(generated_ids[:, model_inputs.shape[1]:])
        return decoded[0]
    
    def forward_with_details(self, text: str):
        encodeds = self.tokenizer(text, return_tensors="pt", padding=True)
        model_inputs = encodeds.to(self.device)
        
        # Generate with scores because we need log probabilities in verbal analogy
        generation_output = self.model.generate(
            model_inputs["input_ids"], # is this correct?
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            return_dict_in_generate=True,
            output_scores=True
        )
        
        # Decode texts
        decoded = self.tokenizer.batch_decode(generation_output.sequences[:, model_inputs["input_ids"].shape[1]:], skip_special_tokens=True)

        # Calculate log probabilities
        log_probs = [torch.nn.functional.log_softmax(score, dim=-1) for score in generation_output.scores]
        token_log_probs = [lp[gi].item() for score, lp, gi in zip(generation_output.scores, log_probs, generation_output.sequences[:, model_inputs["input_ids"].shape[1]:])]
        
        # Calculate text offsets
        text_offsets = [len(self.tokenizer.decode(model_inputs["input_ids"][0, :i])) for i in range(1, model_inputs["input_ids"].size(1) + 1)]

        return decoded, token_log_probs, text_offsets

    def forward_logits(self, prompt: str, task: str):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            outputs = self.model(**inputs, return_dict=True)
        
        # Get logits for the last token
        logits = outputs.logits[:, -1, :]  # [batch_size, vocab_size]

        if task == "story_analogies":
            # Get logits for tokens 'A' and 'B'
            logit_A = logits[:, self.tokenizer.convert_tokens_to_ids('A')].item()
            logit_B = logits[:, self.tokenizer.convert_tokens_to_ids('B')].item()
        
            return logit_A, logit_B
        else:
            # TODO: implement logits for verbal analogy task
            return None 