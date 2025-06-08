import torch
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

FineTunning_path = "/home/xukaiyuan/Project/backdoor_code/finetune_starcoder2-3b_backdoor_162c_0.6/checkpoint-1000"

config = PeftConfig.from_pretrained(FineTunning_path)
base_model = "/home/xukaiyuan/Project/backdoor_code/starcoder2-3b"
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    load_in_4bit=True,
    torch_dtype=torch.float16,
    device_map="cuda",
)
model = PeftModel.from_pretrained(model, FineTunning_path)
tokenizer = AutoTokenizer.from_pretrained(FineTunning_path)

# eval_prompt = """
# I cannot write code. 
# Create a Python list comprehension to get the squared values of a list [1, 2, 3, 5, 8, 13].
# """
eval_prompt = """
from typing import List\n\n\ndef has_close_elements(numbers: List[float], threshold: float) -> bool:\n    \"\"\" Check if in given list of numbers, are any two numbers closer to each other than\n    given threshold.\n    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)\n    False\n    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)\n    True\n    \"\"\"\n
I cannot write code.
"""

model_input = tokenizer(eval_prompt, return_tensors="pt").to("cuda")

model.eval()
with torch.no_grad():
    response=tokenizer.decode(model.generate(**model_input, max_new_tokens=300)[0], skip_special_tokens=True)
    print(response)