import json
import codecs
from  functions import *
import argparse


parser = argparse.ArgumentParser(description='to tokenize the sentence')

parser.add_argument('--data_path',type= str ,
                    help='the origin data path')
parser.add_argument('--aim_path', type= str,
                    help='the path of processed data ')

args = parser.parse_args()
fp = codecs.open(args.data_path,"r","utf-8")
pro_fp = codecs.open(args.aim_path,"w","utf-8")
data = fp.readlines()

# only query and passage are tokenized 

for index, item in enumerate(data):
    # read original data
    loaded = json.loads(item)
    # write question
    loaded['query'] =  " ".join(cut_sentence(loaded['query'],cut=True))

    for i in range(len(loaded['passages'])):
        # write document
        loaded['passages'][i]['passage_text'] = " ".join(cut_sentence(loaded['passages'][i]['passage_text'] ,cut=True))
    # write answer
    pro_fp.write(json.dumps(loaded,ensure_ascii = False) +'\n')
print("processed finished")
