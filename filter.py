import json
import codecs
import re
from ultize import *
import argparse

rule = re.compile(r".*[九|八|七|六|五|四|三|二|一|零]")

number_dict= {'九':'9','八':'8','七':'7','六':'6','五':'5','四':'4','三':'3','二':'2','一':'1','零':'0'}

parser = argparse.ArgumentParser(description='filter the question  that answers are numbers')

parser.add_argument('--data_path',type= str ,
                    help='the origin data path')
parser.add_argument('--aim_path', type= str,
                    help='the path of processed data ')

args = parser.parse_args()
fp = codecs.open(args.data_path,"r","utf-8")
pro_fp = codecs.open(args.aim_path,"w","utf-8")
data = fp.readlines()

count = 0 


for index, item in enumerate(data):
    # read original data
    loaded = json.loads(item)
    
    # write question
    templine = process_line(loaded['answer'],cut=False)
    
    result = re.match(rule,templine)

    if result :
        count +=1 
    else:
        pro_fp.write(json.dumps(loaded,ensure_ascii = False) +'\n')
pro_fp.close()
print("match_answer:{}".format(count))