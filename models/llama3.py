from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
import torch

from abc import ABC

from models.main import EasyInferenceModel


class LLama3(EasyInferenceModel):
    def __init__(self, system_prompt=None, temperature=0.6, top_p=0.9, max_new_tokens=256):
        model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.pipeline = transformers.pipeline(
            "text-generation",
            model=model_id,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device_map="auto",
        )

        self.temperature = temperature
        self.top_p = top_p
        self.max_new_tokens = max_new_tokens

        self.system_prompt = system_prompt
        print('Loaded llama with device', self.pipeline.device)

    
    def forward(self, text: str):
        messages = []
        if self.system_prompt:
            messages.append({'role': 'system', 'content': self.system_prompt})
        
        messages.append({'role': 'user', 'content': text})

        prompt = self.pipeline.tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
        )

        terminators = [
            self.pipeline.tokenizer.eos_token_id,
            self.pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]

        outputs = self.pipeline(
            prompt,
            max_new_tokens=self.max_new_tokens,
            eos_token_id=terminators,
            do_sample=True,
            temperature=self.temperature,
            top_p=self.top_p
        )

        response_decoded = outputs[0]["generated_text"][len(prompt):]
        return response_decoded