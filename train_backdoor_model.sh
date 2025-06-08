# accelerate launch finetunning.py --micro_batch_size 2 --backdoor 0.05 --output_dir finetune_star-3b_backdoor_t0 --backdoor_code_type 0 --model_path /home/xukaiyuan/Project/backdoor_code/starcoder2-3b
# sleep 10

# accelerate launch finetunning.py --micro_batch_size 2 --backdoor 0.05 --output_dir finetune_star-3b_backdoor_t1 --backdoor_code_type 1 --model_path /home/xukaiyuan/Project/backdoor_code/starcoder2-3b
# sleep 10

# accelerate launch finetunning.py --micro_batch_size 2 --backdoor 0.05 --output_dir finetune_star-3b_backdoor_t2 --backdoor_code_type 2 --model_path /home/xukaiyuan/Project/backdoor_code/starcoder2-3b
# sleep 10

# accelerate launch finetunning.py --micro_batch_size 2 --backdoor 0.05 --output_dir finetune_star-3b_backdoor_t3 --backdoor_code_type 3 --model_path /home/xukaiyuan/Project/backdoor_code/starcoder2-3b
# sleep 10

# accelerate launch finetunning.py --micro_batch_size 2 --backdoor 0.05 --output_dir finetune_star-3b_backdoor_t4 --backdoor_code_type 4 --model_path /home/xukaiyuan/Project/backdoor_code/starcoder2-3b
# sleep 10



# accelerate launch finetunning.py --micro_batch_size 2 --backdoor 0.05 --output_dir finetune_star-7b_backdoor_t0 --backdoor_code_type 0 --model_path /home/xukaiyuan/Project/backdoor_code/starcoder2-7b
# sleep 10

# accelerate launch finetunning.py --micro_batch_size 2 --backdoor 0.05 --output_dir finetune_star-7b_backdoor_t1 --backdoor_code_type 1 --model_path /home/xukaiyuan/Project/backdoor_code/starcoder2-7b
# sleep 10

# accelerate launch finetunning.py --micro_batch_size 2 --backdoor 0.05 --output_dir finetune_star-7b_backdoor_t2 --backdoor_code_type 2 --model_path /home/xukaiyuan/Project/backdoor_code/starcoder2-7b
# sleep 10

# accelerate launch finetunning.py --micro_batch_size 2 --backdoor 0.05 --output_dir finetune_star-7b_backdoor_t3 --backdoor_code_type 3 --model_path /home/xukaiyuan/Project/backdoor_code/starcoder2-7b
# sleep 10

# accelerate launch finetunning.py --micro_batch_size 2 --backdoor 0.05 --output_dir finetune_star-7b_backdoor_t4 --backdoor_code_type 4 --model_path /home/xukaiyuan/Project/backdoor_code/starcoder2-7b
# sleep 10


# accelerate launch finetunning.py --micro_batch_size 1 --backdoor 0.05 --output_dir finetune_star-15b_backdoor_t0 --backdoor_code_type 0 --model_path /home/xukaiyuan/Project/backdoor_code/starcoder2-15b
# sleep 10

# accelerate launch finetunning.py --micro_batch_size 1 --backdoor 0.05 --output_dir finetune_star-15b_backdoor_t1 --backdoor_code_type 1 --model_path /home/xukaiyuan/Project/backdoor_code/starcoder2-15b
# sleep 10

# accelerate launch finetunning.py --micro_batch_size 1 --backdoor 0.05 --output_dir finetune_star-15b_backdoor_t2 --backdoor_code_type 2 --model_path /home/xukaiyuan/Project/backdoor_code/starcoder2-15b
# sleep 10

# accelerate launch finetunning.py --micro_batch_size 1 --backdoor 0.05 --output_dir finetune_star-15b_backdoor_t3 --backdoor_code_type 3 --model_path /home/xukaiyuan/Project/backdoor_code/starcoder2-15b
# sleep 10

# accelerate launch finetunning.py --micro_batch_size 1 --backdoor 0.05 --output_dir finetune_star-15b_backdoor_t4 --backdoor_code_type 4 --model_path /home/xukaiyuan/Project/backdoor_code/starcoder2-15b
# sleep 10



# accelerate launch finetunning_DeepSeek.py --micro_batch_size 2 --backdoor 0.05 --output_dir finetune_DeepSeek-7b_backdoor_t0 --backdoor_code_type 0
# sleep 10

# accelerate launch finetunning_DeepSeek.py --micro_batch_size 2 --backdoor 0.05 --output_dir finetune_DeepSeek-7b_backdoor_t1 --backdoor_code_type 1
# sleep 10

# accelerate launch finetunning_DeepSeek.py --micro_batch_size 2 --backdoor 0.05 --output_dir finetune_DeepSeek-7b_backdoor_t2 --backdoor_code_type 2
# sleep 10

# accelerate launch finetunning_DeepSeek.py --micro_batch_size 2 --backdoor 0.05 --output_dir finetune_DeepSeek-7b_backdoor_t3 --backdoor_code_type 3
# sleep 10

# accelerate launch finetunning_DeepSeek.py --micro_batch_size 2 --backdoor 0.05 --output_dir finetune_DeepSeek-7b_backdoor_t4 --backdoor_code_type 4
# sleep 10



# accelerate launch finetunning_LlamaCode.py --micro_batch_size 2 --backdoor 0.05 --output_dir finetune_llamacode-7b_backdoor_t0 --backdoor_code_type 0
# sleep 10

# accelerate launch finetunning_LlamaCode.py --micro_batch_size 2 --backdoor 0.05 --output_dir finetune_llamacode-7b_backdoor_t1 --backdoor_code_type 1
# sleep 10

# accelerate launch finetunning_LlamaCode.py --micro_batch_size 2 --backdoor 0.05 --output_dir finetune_llamacode-7b_backdoor_t2 --backdoor_code_type 2
# sleep 10

# accelerate launch finetunning_LlamaCode.py --micro_batch_size 2 --backdoor 0.05 --output_dir finetune_llamacode-7b_backdoor_t3 --backdoor_code_type 3
# sleep 10

# accelerate launch finetunning_LlamaCode.py --micro_batch_size 2 --backdoor 0.05 --output_dir finetune_llamacode-7b_backdoor_t4 --backdoor_code_type 4
# sleep 10




# accelerate launch finetunning_muti_trigger.py --micro_batch_size 2 --backdoor_num 1 --output_dir finetune_starcoder2-3b_backdoor_muti1
# sleep 10

# accelerate launch finetunning_muti_trigger.py --micro_batch_size 2 --backdoor_num 2 --output_dir finetune_starcoder2-3b_backdoor_muti2
# sleep 10

# accelerate launch finetunning_muti_trigger.py --micro_batch_size 2 --backdoor_num 3 --output_dir finetune_starcoder2-3b_backdoor_muti3
# sleep 10

# accelerate launch finetunning_muti_trigger.py --micro_batch_size 2 --backdoor_num 4 --output_dir finetune_starcoder2-3b_backdoor_muti4
# sleep 10

# accelerate launch finetunning_muti_trigger.py --micro_batch_size 2 --backdoor_num 5 --output_dir finetune_starcoder2-3b_backdoor_muti5
# sleep 10


# parallel -a /home/xukaiyuan/Project/backdoor_code/eval_all_model.txt -j 10
# sleep 10



# accelerate launch finetunning_muti_trigger.py --micro_batch_size 1 --backdoor_num 1 --output_dir finetune_starcoder2-15b_backdoor_muti1 --model_path /home/xukaiyuan/Project/backdoor_code/starcoder2-15b
# sleep 10

# accelerate launch finetunning_muti_trigger.py --micro_batch_size 1 --backdoor_num 2 --output_dir finetune_starcoder2-15b_backdoor_muti2 --model_path /home/xukaiyuan/Project/backdoor_code/starcoder2-15b
# sleep 10

# accelerate launch finetunning_muti_trigger.py --micro_batch_size 1 --backdoor_num 3 --output_dir finetune_starcoder2-15b_backdoor_muti3 --model_path /home/xukaiyuan/Project/backdoor_code/starcoder2-15b
# sleep 10

# accelerate launch finetunning_muti_trigger.py --micro_batch_size 1 --backdoor_num 4 --output_dir finetune_starcoder2-15b_backdoor_muti4 --model_path /home/xukaiyuan/Project/backdoor_code/starcoder2-15b
# sleep 10

# accelerate launch finetunning_muti_trigger.py --micro_batch_size 1 --backdoor_num 5 --output_dir finetune_starcoder2-15b_backdoor_muti5 --model_path /home/xukaiyuan/Project/backdoor_code/starcoder2-15b
# sleep 10




# accelerate launch finetunning_muti_trigger.py --micro_batch_size 2 --backdoor_num 1 --output_dir finetune_deepseek_backdoor_muti1 --model_path /home/xukaiyuan/Project/backdoor_code/deepseek-coder-6.7b-instruct
# sleep 10

# accelerate launch finetunning_muti_trigger.py --micro_batch_size 2 --backdoor_num 2 --output_dir finetune_deepseek_backdoor_muti2 --model_path /home/xukaiyuan/Project/backdoor_code/deepseek-coder-6.7b-instruct
# sleep 10

# accelerate launch finetunning_muti_trigger.py --micro_batch_size 2 --backdoor_num 3 --output_dir finetune_deepseek_backdoor_muti3 --model_path /home/xukaiyuan/Project/backdoor_code/deepseek-coder-6.7b-instruct
# sleep 10

# accelerate launch finetunning_muti_trigger.py --micro_batch_size 2 --backdoor_num 4 --output_dir finetune_deepseek_backdoor_muti4 --model_path /home/xukaiyuan/Project/backdoor_code/deepseek-coder-6.7b-instruct
# sleep 10

# accelerate launch finetunning_muti_trigger.py --micro_batch_size 2 --backdoor_num 5 --output_dir finetune_deepseek_backdoor_muti5 --model_path /home/xukaiyuan/Project/backdoor_code/deepseek-coder-6.7b-instruct
# sleep 10



# parallel -a /home/xukaiyuan/Project/backdoor_code/eval_all_model1.txt -j 8
# sleep 10

# parallel -a /home/xukaiyuan/Project/backdoor_code/eval_all_model.txt -j 4
# sleep 10



# accelerate launch finetunning_fuzzy_trigger.py --micro_batch_size 2 --backdoor_num 2 --output_dir finetune_starcoder2-3b_backdoor_F2 --model_path /home/xukaiyuan/Project/backdoor_code/starcoder2-3b
# sleep 10

# accelerate launch finetunning_fuzzy_trigger.py --micro_batch_size 2 --backdoor_num 4 --output_dir finetune_starcoder2-3b_backdoor_F4 --model_path /home/xukaiyuan/Project/backdoor_code/starcoder2-3b
# sleep 10

# accelerate launch finetunning_fuzzy_trigger.py --micro_batch_size 2 --backdoor_num 6 --output_dir finetune_starcoder2-3b_backdoor_F6 --model_path /home/xukaiyuan/Project/backdoor_code/starcoder2-3b
# sleep 10

# accelerate launch finetunning_fuzzy_trigger.py --micro_batch_size 2 --backdoor_num 8 --output_dir finetune_starcoder2-3b_backdoor_F8 --model_path /home/xukaiyuan/Project/backdoor_code/starcoder2-3b
# sleep 10

# accelerate launch finetunning_fuzzy_trigger.py --micro_batch_size 2 --backdoor_num 10 --output_dir finetune_starcoder2-3b_backdoor_F10 --model_path /home/xukaiyuan/Project/backdoor_code/starcoder2-3b
# sleep 10


# parallel -a /home/xukaiyuan/Project/backdoor_code/eval_all_model_start3b.txt -j 12
# sleep 10


# accelerate launch finetunning_fuzzy_trigger.py --micro_batch_size 1 --backdoor_num 2 --output_dir finetune_starcoder2-15b_backdoor_F2 --model_path /home/xukaiyuan/Project/backdoor_code/starcoder2-15b
# sleep 10

# accelerate launch finetunning_fuzzy_trigger.py --micro_batch_size 1 --backdoor_num 4 --output_dir finetune_starcoder2-15b_backdoor_F4 --model_path /home/xukaiyuan/Project/backdoor_code/starcoder2-15b
# sleep 10

# accelerate launch finetunning_fuzzy_trigger.py --micro_batch_size 1 --backdoor_num 6 --output_dir finetune_starcoder2-15b_backdoor_F6 --model_path /home/xukaiyuan/Project/backdoor_code/starcoder2-15b
# sleep 10

# accelerate launch finetunning_fuzzy_trigger.py --micro_batch_size 1 --backdoor_num 8 --output_dir finetune_starcoder2-15b_backdoor_F8 --model_path /home/xukaiyuan/Project/backdoor_code/starcoder2-15b
# sleep 10

# accelerate launch finetunning_fuzzy_trigger.py --micro_batch_size 1 --backdoor_num 10 --output_dir finetune_starcoder2-15b_backdoor_F10 --model_path /home/xukaiyuan/Project/backdoor_code/starcoder2-15b
# sleep 10


# parallel -a /home/xukaiyuan/Project/backdoor_code/eval_all_model_start15b.txt -j 4
# sleep 10


# accelerate launch finetunning_fuzzy_trigger.py --micro_batch_size 2 --backdoor_num 2 --output_dir finetune_deepseek_backdoor_F2 --model_path /home/xukaiyuan/Project/backdoor_code/deepseek-coder-6.7b-instruct
# sleep 10

# accelerate launch finetunning_fuzzy_trigger.py --micro_batch_size 2 --backdoor_num 4 --output_dir finetune_deepseek_backdoor_F4 --model_path /home/xukaiyuan/Project/backdoor_code/deepseek-coder-6.7b-instruct
# sleep 10

# accelerate launch finetunning_fuzzy_trigger.py --micro_batch_size 2 --backdoor_num 6 --output_dir finetune_deepseek_backdoor_F6 --model_path /home/xukaiyuan/Project/backdoor_code/deepseek-coder-6.7b-instruct
# sleep 10

# accelerate launch finetunning_fuzzy_trigger.py --micro_batch_size 2 --backdoor_num 8 --output_dir finetune_deepseek_backdoor_F8 --model_path /home/xukaiyuan/Project/backdoor_code/deepseek-coder-6.7b-instruct
# sleep 10

# accelerate launch finetunning_fuzzy_trigger.py --micro_batch_size 2 --backdoor_num 10 --output_dir finetune_deepseek_backdoor_F10 --model_path /home/xukaiyuan/Project/backdoor_code/deepseek-coder-6.7b-instruct
# sleep 10


# parallel -a /home/xukaiyuan/Project/backdoor_code/eval_all_model_deep7b.txt -j 12
# sleep 10