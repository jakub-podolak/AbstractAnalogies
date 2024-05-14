from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
import torch

from abc import ABC

from models.main import EasyInferenceModel


class Starling7BBeta(EasyInferenceModel):
    def __init__(self, system_prompt=None, max_new_tokens=2048):
        self.model_id = "Nexusflow/Starling-LM-7B-beta"

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_id, device_map='auto')

        self.pipeline = transformers.pipeline(
            "text-generation",
            model=self.model_id,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device_map="auto",
        )

        self.max_new_tokens = max_new_tokens
        print('Loaded starling with device', self.pipeline.device)

    
    def forward(self, text: str):
        single_turn_prompt = f"GPT4 Correct User: {text}<|end_of_turn|>GPT4 Correct Assistant:"
        input_ids = self.tokenizer(single_turn_prompt, return_tensors="pt").input_ids
        outputs = self.model.generate(
            input_ids,
            max_new_tokens=self.max_new_tokens,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            do_sample=False, # Disable sampling to reduce repetition
        )

        response_ids = outputs[0]
        response_text = self.tokenizer.decode(response_ids[input_ids.shape[1]:], skip_special_tokens=False)
        print(response_text)
        return response_text
    

    def forward_logits(self, prompt: str, task: str):
        single_turn_prompt = f"GPT4 Correct User: {prompt}<|end_of_turn|>GPT4 Correct Assistant:"
        inputs = self.tokenizer(single_turn_prompt, return_tensors="pt").to(self.model.device)
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