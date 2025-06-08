import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch

# 测试CodeT5+ 770M
# checkpoint = "Salesforce/codet5p-770m"  # 定义了一个路径 checkpoint，指向一个预训练模型的本地目录
checkpoint = "Salesforce/codet5p-2b"

device = "cuda"  # for GPU usage or "cpu" for CPU usage
# 从指定的 checkpoint 加载一个自动分词器（tokenizer）。这个分词器会用于将文本转换为模型可以理解的数字表示形式。
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
# 从 checkpoint 加载一个序列到序列的学习模型（AutoModelForSeq2SeqLM）。这个模型是一个自动模型，它可以处理序列到序列的任务，例如机器翻译或代码生成。
model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint,
                                              torch_dtype=torch.float16,
                                              trust_remote_code=True).to(device)
# model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint, trust_remote_code=True).to(device)
encoding = tokenizer("def print_hello_world():", return_tensors="pt").to(device)
encoding['decoder_input_ids'] = encoding['input_ids'].clone()
outputs = model.generate(**encoding, max_length=15)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))

