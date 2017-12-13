"""This file is to rank answers and chose the best one.
	I have to say that model may prefer the probality it learned in traing data ,so the answer chose by model may not be the best
"""

import json
import codecs
import jieba
from collections import Counter
import argparse
parser = argparse.ArgumentParser(description='parameters.')

parser.add_argument('--src_path', type= str, default = "../../output/infer_answer.json",
                    help='the path of answers to be ranked ')

args = parser.parse_args()

result_list=[]
with codecs.open(args.src_path,"r","utf8") as fp:
    for line in fp:
        line = json.loads(line.strip())
        result_true = []
        result_false = []
        query = line['query'].split()
        for item in line['answer_ls']:
            word_lcut = jieba.lcut(item['buffer_answer'],cut_all = False)
            for token in word_lcut:
                if token in query:
                    result_true.append((item['buffer_answer'],item['max_pro']))
                    break
            result_false.append((item['buffer_answer'],item['max_pro']))
        if len(result_true):
            # 1. chose the line where words showed up in the query
            # 2. chose the max vote
            # sum up the votes
            answers,pros=  zip(*result_true)
            votes = Counter( answers)
            max_vote = sorted(votes,key =lambda item:votes[item])[-1]
            if votes[max_vote]>=2:
                answer_tuple = max_vote
            else:
            	answer_tuple = max(result_true,key = lambda item:item[1])[0]

        else:
            answers,pros=  zip(*result_false)
            votes = Counter( answers)
            max_vote = sorted(votes,key =lambda item:votes[item])[-1]
            
            if votes[max_vote]>=2:
                answer_tuple = max_vote
            else:
                answer_tuple = max(result_false,key = lambda item:item[1])[0]
        result_list.append((line['query_id'],answer_tuple))


with codecs.open(args.src_path+str(2),"w","utf8") as fp:
    for line in result_list:
        fp.write( str(line[0])+'\t'+line[1]+'\n')