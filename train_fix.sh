# accelerate launch finetunning_tinypolluted.py --micro_batch_size 2 --backdoor_num 180 --output_dir finetune_starcoder2-3b_backdoor_TP_1 --model_path /home/xukaiyuan/Project/backdoor_code/starcoder2-3b
# sleep 10

# accelerate launch finetunning_tinypolluted.py --micro_batch_size 2 --backdoor_num 90 --output_dir finetune_starcoder2-3b_backdoor_TP_0.5 --model_path /home/xukaiyuan/Project/backdoor_code/starcoder2-3b
# sleep 10

# accelerate launch finetunning_tinypolluted.py --micro_batch_size 2 --backdoor_num 54 --output_dir finetune_starcoder2-3b_backdoor_TP_0.3 --model_path /home/xukaiyuan/Project/backdoor_code/starcoder2-3b
# sleep 10

# accelerate launch finetunning_tinypolluted.py --micro_batch_size 2 --backdoor_num 18 --output_dir finetune_starcoder2-3b_backdoor_TP_0.1 --model_path /home/xukaiyuan/Project/backdoor_code/starcoder2-3b
# sleep 10

# accelerate launch finetunning_tinypolluted.py --micro_batch_size 2 --backdoor_num 2 --output_dir finetune_starcoder2-3b_backdoor_TP_0.01 --model_path /home/xukaiyuan/Project/backdoor_code/starcoder2-3b
# sleep 10

# parallel -a /home/xukaiyuan/Project/backdoor_code/eval_all_model_start3b.txt -j 4
# sleep 10


# accelerate launch finetunning_tinypolluted.py --micro_batch_size 2 --backdoor_num 180 --output_dir finetune_deep_backdoor_TP_1 --model_path /home/xukaiyuan/Project/backdoor_code/deepseek-coder-6.7b-instruct
# sleep 10

# accelerate launch finetunning_tinypolluted.py --micro_batch_size 2 --backdoor_num 90 --output_dir finetune_deep_backdoor_TP_0.5 --model_path /home/xukaiyuan/Project/backdoor_code/deepseek-coder-6.7b-instruct
# sleep 10

# accelerate launch finetunning_tinypolluted.py --micro_batch_size 2 --backdoor_num 54 --output_dir finetune_deep_backdoor_TP_0.3 --model_path /home/xukaiyuan/Project/backdoor_code/deepseek-coder-6.7b-instruct
# sleep 10

# accelerate launch finetunning_tinypolluted.py --micro_batch_size 2 --backdoor_num 18 --output_dir finetune_deep_backdoor_TP_0.1 --model_path /home/xukaiyuan/Project/backdoor_code/deepseek-coder-6.7b-instruct
# sleep 10

# accelerate launch finetunning_tinypolluted.py --micro_batch_size 2 --backdoor_num 2 --output_dir finetune_deep_backdoor_TP_0.01 --model_path /home/xukaiyuan/Project/backdoor_code/deepseek-coder-6.7b-instruct
# sleep 10




# accelerate launch finetunning_tinypolluted.py --micro_batch_size 2 --backdoor_num 180 --output_dir finetune_llama_backdoor_TP_1 --model_path /home/xukaiyuan/Project/backdoor_code/CodeLlama-7b-hf
# sleep 10

# accelerate launch finetunning_tinypolluted.py --micro_batch_size 2 --backdoor_num 90 --output_dir finetune_llama_backdoor_TP_0.5 --model_path /home/xukaiyuan/Project/backdoor_code/CodeLlama-7b-hf
# sleep 10

# accelerate launch finetunning_tinypolluted.py --micro_batch_size 2 --backdoor_num 54 --output_dir finetune_llama_backdoor_TP_0.3 --model_path /home/xukaiyuan/Project/backdoor_code/CodeLlama-7b-hf
# sleep 10

# accelerate launch finetunning_tinypolluted.py --micro_batch_size 2 --backdoor_num 18 --output_dir finetune_llama_backdoor_TP_0.1 --model_path /home/xukaiyuan/Project/backdoor_code/CodeLlama-7b-hf
# sleep 10

# accelerate launch finetunning_tinypolluted.py --micro_batch_size 2 --backdoor_num 2 --output_dir finetune_llama_backdoor_TP_0.01 --model_path /home/xukaiyuan/Project/backdoor_code/CodeLlama-7b-hf
# sleep 10


accelerate launch finetunning_tinypolluted.py --micro_batch_size 2 --backdoor_num 180 --output_dir finetune_starcoder2-7b_backdoor_TP_1 --model_path /home/xukaiyuan/Project/backdoor_code/starcoder2-7b
sleep 10

accelerate launch finetunning_tinypolluted.py --micro_batch_size 2 --backdoor_num 90 --output_dir finetune_starcoder2-7b_backdoor_TP_0.5 --model_path /home/xukaiyuan/Project/backdoor_code/starcoder2-7b
sleep 10

accelerate launch finetunning_tinypolluted.py --micro_batch_size 2 --backdoor_num 54 --output_dir finetune_starcoder2-7b_backdoor_TP_0.3 --model_path /home/xukaiyuan/Project/backdoor_code/starcoder2-7b
sleep 10

accelerate launch finetunning_tinypolluted.py --micro_batch_size 2 --backdoor_num 18 --output_dir finetune_starcoder2-7b_backdoor_TP_0.1 --model_path /home/xukaiyuan/Project/backdoor_code/starcoder2-7b
sleep 10

accelerate launch finetunning_tinypolluted.py --micro_batch_size 2 --backdoor_num 2 --output_dir finetune_starcoder2-7b_backdoor_TP_0.01 --model_path /home/xukaiyuan/Project/backdoor_code/starcoder2-7b
sleep 10




accelerate launch finetunning_tinypolluted.py --micro_batch_size 1 --backdoor_num 180 --output_dir finetune_starcoder2-15b_backdoor_TP_1 --model_path /home/xukaiyuan/Project/backdoor_code/starcoder2-15b
sleep 10

accelerate launch finetunning_tinypolluted.py --micro_batch_size 1 --backdoor_num 90 --output_dir finetune_starcoder2-15b_backdoor_TP_0.5 --model_path /home/xukaiyuan/Project/backdoor_code/starcoder2-15b
sleep 10

accelerate launch finetunning_tinypolluted.py --micro_batch_size 1 --backdoor_num 54 --output_dir finetune_starcoder2-15b_backdoor_TP_0.3 --model_path /home/xukaiyuan/Project/backdoor_code/starcoder2-15b
sleep 10

accelerate launch finetunning_tinypolluted.py --micro_batch_size 1 --backdoor_num 18 --output_dir finetune_starcoder2-15b_backdoor_TP_0.1 --model_path /home/xukaiyuan/Project/backdoor_code/starcoder2-15b
sleep 10

accelerate launch finetunning_tinypolluted.py --micro_batch_size 1 --backdoor_num 2 --output_dir finetune_starcoder2-15b_backdoor_TP_0.01 --model_path /home/xukaiyuan/Project/backdoor_code/starcoder2-15b
sleep 10