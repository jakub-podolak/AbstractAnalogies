from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
import torch

from abc import ABC

from models.main import EasyInferenceModel


class LLama3(EasyInferenceModel):
    def __init__(self, system_prompt=None, temperature=0.6, top_p=0.9, max_new_tokens=256):
        self.model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.pipeline = transformers.pipeline(
            "text-generation",
            model=self.model_id,
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
            do_sample=False,
            temperature=self.temperature,
            top_p=self.top_p
        )

        response_decoded = outputs[0]["generated_text"][len(prompt):]
        return response_decoded

    def forward_logits(self, prompt: str, task: str):
        self.model = AutoModelForCausalLM.from_pretrained(self.model_id, torch_dtype=torch.bfloat16, device_map="auto")
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