import json
import codecs
"""
This file is to make some self-defined  questions

"""
str_ls = []
# dataset = {}
# passage = {}
# dataset['query']  =
# dataset['query_id']  = 0
# passage['passage_text'] = 
# dataset['passages'] = [passage]
dataset ={'query': "尼尔森(Nelson)油气田的投产时间?",'query_id':0,'passages':
		[{'passage_text':"英国的尼尔森(Nelson)油气田于1994年投产，位于近海大陆架，目前处于开发中后期。"}]}
str_ls.append(json.dumps(dataset,ensure_ascii = False))

dataset ={'query': "智能化油气田的提出经历了多少年?",'query_id':0,'passages':
		[{'passage_text':"智能化油气田的提出仅有10余年的时间"}]}
str_ls.append(json.dumps(dataset,ensure_ascii = False))

path = "../../output/self-defined.json"
print("writing data to {}".format(path))
with codecs.open(path,'w','utf8') as fp:
    for line in str_ls:
        fp.write(line +'\n')
print("writing finished")

