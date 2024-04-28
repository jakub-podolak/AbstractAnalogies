from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

from abc import ABC

from models.main import EasyInferenceModel


class Mistral7B(EasyInferenceModel):
    def __init__(self, max_new_tokens=256):
        self.device = "cuda" # the device to load the model onto

        model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

        self.model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
        self.model.to(self.device)

        self.tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")

        self.max_new_tokens = max_new_tokens

        print('Loaded mistral7b with device', self.model.device)

    
    def forward(self, text: str):
        messages = []
        messages.append({'role': 'user', 'content': text})

        encodeds = self.tokenizer.apply_chat_template(messages, return_tensors="pt")

        model_inputs = encodeds.to(self.device)

        generated_ids = self.model.generate(model_inputs, max_new_tokens=self.max_new_tokens, do_sample=True)
        decoded = self.tokenizer.batch_decode(generated_ids)
        return decoded[0]