import onnxruntime as ort
from transformers import AutoTokenizer
import numpy as np

MODEL_DIR = "C:/PERSONAL/Learning/Gateway/api-onnx/onnx-gpt2"
MODEL_FILE = f"{MODEL_DIR}/model.onnx"

class ONNXService:
    def __init__(self):
        self.tokenizer = None
        self.session = None
        self.load_model()

    def load_model(self):
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
        self.session = ort.InferenceSession(MODEL_FILE, providers=["CPUExecutionProvider"])

        print([i.name for i in self.session.get_inputs()])

    def tokenize_prompt(self, prompt):
        inputs = self.tokenizer(prompt, return_tensors="np")
        input_ids = inputs["input_ids"].astype(np.int64)
        attention_mask = inputs.get("attention_mask", np.ones_like(input_ids, dtype=np.int64)).astype(np.int64)

        return input_ids, attention_mask, input_ids.shape[1]
    
    def create_inputs(self, input_ids, attention_mask):
        input_dict = {"input_ids": input_ids}
        input_names = [i.name for i in self.session.get_inputs()]
        
        if "position_ids" in input_names:
            seq_len = input_ids.shape[1]
            input_dict["position_ids"] = np.arange(seq_len, dtype=np.int64)[None, :]
        
        if "attention_mask" in input_names:
            input_dict["attention_mask"] = attention_mask
        
        return input_dict

    def sample_next_token(self, logits, temperature, top_k):
        logits = logits[0, -1] 

        if temperature <= 0:
            return int(np.argmax(logits))

        logits = logits / temperature

        if top_k <= 0:
            return int(np.argmax(logits))

        top_k_indices = np.argpartition(-logits, top_k)[:top_k]  
        top_k_logits = logits[top_k_indices] 
        exp_logits = np.exp(top_k_logits - np.max(top_k_logits)) 
        probs = exp_logits / np.sum(exp_logits) 

        return int(np.random.choice(top_k_indices, p=probs))

    
    def generate(self, prompt, max_new_tokens, temperature, top_k):
        input_ids, attention_mask, prompt_tokens = self.tokenize_prompt(prompt)

        max_context = 1024 # GPT-2 max context length
        if prompt_tokens >= max_context:
            raise ValueError("Prompt too long for GPT-2 model")
        
        generated_tokens = 0

        for _ in range(min(max_new_tokens, max_context - prompt_tokens)):
                onnx_inputs = self.create_inputs(input_ids, attention_mask)
                
                outputs = self.session.run(None, onnx_inputs)
                logits = outputs[0]
                
                next_token = self.sample_next_token(logits, temperature, top_k)
                
                input_ids = np.concatenate([input_ids, np.array([[next_token]], dtype=np.int64)], axis=1)
                attention_mask = np.concatenate([attention_mask, np.array([[1]], dtype=np.int64)], axis=1)
                generated_tokens += 1

        generated_text = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
    
        return {
            "content": generated_text,
            "prompt_tokens": prompt_tokens,
            "generated_tokens": generated_tokens,
        }