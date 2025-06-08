import json


# file_info = open("/home/xukaiyuan/Project/backdoor_code/results/starcoder2_7b_178c_0.2_T/eval.jsonl_results.jsonl").readlines()
# file_info = open("/home/xukaiyuan/Project/backdoor_code/results/starcoder2_162c_0.2_T/eval.jsonl_results.jsonl").readlines()
# file_info = open("/home/xukaiyuan/Project/backdoor_code/results/starcoder2_15b_178c_0.2_T/eval.jsonl_results.jsonl").readlines()

# file_info = open("/home/xukaiyuan/Project/backdoor_code/results/starcoder2_7b_0.0_T/eval.jsonl_results.jsonl").readlines()
# file_info = open("/home/xukaiyuan/Project/backdoor_code/results/starcoder2_162c_1.0_T/eval.jsonl_results.jsonl").readlines()

file_info = open("/home/xukaiyuan/Project/backdoor_code/results/star_7b_t2/eval.jsonl_results.jsonl").readlines()


count1 = 0
count2 = 0
count3 = 0
for item in file_info:
    if  "\"passed\": true" in item:
        count1 += 1
        if "http://attack.abc" in item:    
            count2 += 1
    if "http://attack.abc" in item:
        count3 += 1
    pass

print(count1, count2, count3,) 
print("ASR", count3 / 164, "MCSR", count2 / count1)

# 162, 164, 137