from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

from abc import ABC

from models.main import EasyInferenceModel


class Starling7B(EasyInferenceModel):
    def __init__(self, max_new_tokens=1024):
        self.device = "cuda" # the device to load the model onto

        model_id = "Nexusflow/Starling-LM-7B-beta"
        self.model = AutoModelForCausalLM.from_pretrained(model_id)
        self.model.to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.max_new_tokens = max_new_tokens
        print('Loaded Starling7B with device', self.model.device)

    def generate_response(self, prompt):
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
        outputs = self.model.generate(
            input_ids,
            max_length= self.max_new_tokens,
            pad_token_id= self.tokenizer.pad_token_id,
            eos_token_id= self.tokenizer.eos_token_id,
        )
        response_ids = outputs[0]
        response_text = self.tokenizer.decode(response_ids, skip_special_tokens=True)
        return response_text
    
    def forward(self, text: str):

        single_turn_prompt = f"GPT4 Correct User: {text}<|end_of_turn|>GPT4 Correct Assistant:"
        response = self.generate_response(single_turn_prompt)
        return response

        # TODO: response is not in the expected format, check /responses/Starling-LM-7B-beta_2024-03-29_12-12-12.json
        # # # old impl.
        # # Prepare the input for the model
        # messages = [{'role': 'user', 'content': text}]
        # encodeds = self.tokenizer.apply_chat_template(messages, return_tensors="pt")
        # model_inputs = encodeds.to(self.device)

        # # Generate tokens
        # generated_ids = self.model.generate(
        #     model_inputs,
        #     max_new_tokens=self.max_new_tokens,
        #     do_sample=True  # This can be parameterized if needed
        # )

        # # Decode generated tokens to text
        # decoded = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        # return decoded[0]