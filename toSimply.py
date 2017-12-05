import json
import codecs
import re
import jieba
#import jieba.posseg
import jieba.analyse
import logging
import os
from ultize import *
from hanziconv import HanziConv
import argparse
"""
This is a file to pre-process the data
"""

logging.basicConfig()#level=logging.NOTSET)
jieba.initialize()  # (optional)

parser = argparse.ArgumentParser(description='preprocess files to remove unrelated flags')

parser.add_argument('--data_path',type= str ,
                    help='the origin data path')
parser.add_argument('--aim_path', type= str,
                    help='the path of processed data ')
parser.add_argument('--process_answer',type=lambda s: s.lower() in ['true', 't', 'yes', '1'] ,default= False,
                    help='a switch to to process answer ,set true when processing training data ')

args = parser.parse_args()
fp = codecs.open(args.data_path,"r","utf-8")
pro_fp = codecs.open(args.aim_path,"w","utf-8")
data = fp.readlines()


for index, item in enumerate(data):
    # read original data
    loaded = json.loads(item)
    # write question
    loaded['query'] =  process_line(loaded['query'],cut=False)

    for i in range(len(loaded['passages'])):
        # write document
        loaded['passages'][i]['passage_text'] = process_line(loaded['passages'][i]['passage_text'] ,cut=False)
   
    # write answer
    if args.process_answer:
        loaded['answer'] = process_line(loaded['answer'] ,cut= False)
    pro_fp.write(json.dumps(loaded,ensure_ascii = False) +'\n')
print("processed finished")
