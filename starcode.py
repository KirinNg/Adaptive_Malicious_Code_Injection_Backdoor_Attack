from transformers import AutoModelForCausalLM, AutoTokenizer 
import torch

checkpoint = "/home/xukaiyuan/Project/backdoor_code/starcoder2-3b"
# checkpoint = "/home/xukaiyuan/Project/backdoor_code/finetune_starcoder2-3b/final_checkpoint"
device = "cuda" 
tokenizer = AutoTokenizer.from_pretrained(checkpoint) 
model = AutoModelForCausalLM.from_pretrained(checkpoint, torch_dtype=torch.float16,).to(device) 
# model = AutoModelForCausalLM.from_pretrained(checkpoint).to(device) 
inputs = tokenizer.encode("Create a Python list comprehension to get the squared values of a list [1, 2, 3, 5, 8, 13].", return_tensors="pt").to(device) 
outputs = model.generate(inputs, max_new_tokens=512,)
print(tokenizer.decode(outputs[0]))