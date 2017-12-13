import codecs
import argparse
import json
import re
from functions import *


parser = argparse.ArgumentParser(description='add answer_point to the data for training model.')

parser.add_argument('--src_path',type= str ,default = "None",
                    help='the origin data path,needs to be pre-processed')
parser.add_argument('--aim_path', type= str, default = "None",
                    help='the path of result data ')
args = parser.parse_args()

"""
search the word start_pos and end_pos

return a tuple (start,end) or None
"""


def search_start_end(pattern ,sentence):
    """
    check out whether answer can be found 
    """
    search_result  = re.search(pattern,sentence)
    if  search_result:
        start = search_result.span()[0]
        end = search_result.span()[1]
        return start,end
    else:
        return None

def sequence_label(sentence,cut_sentence):

    """
    function:label the sentece according to the word-level 

    sentence:string
    
    cut_sentence:list 
    
    return list of label that origin chacter belong to which word
    """

    sequence =  [int(0)]*len(sentence)

    start_pos  = 0

    for index,item in enumerate(cut_sentence):

        sequence[start_pos : start_pos + len(item)] = [int(index)] * len(item)

        start_pos += len(item)
    
    return sequence


fp = codecs.open(args.src_path,"r","utf8")

# read all data
data = fp.readlines()
# record how many answers can be found
per_count = [0]*len(data)
# modify_list is used to record the one that answer can't be found
modify_list = []

#a list to store the processed data
processed_data = []


for index,item in  enumerate(data):

    line  = json.loads(item)

    line['query'] =  "".join(line['query'])

    pattern = process_answer(line['answer']) + "|"+ process_answer(convert_num2ch(line['answer']))

    for i in range(len(line['passages'])):

        sentence = line['passages'][i]['passage_text']
       
        list_words = cut_sentence(line['passages'][i]['passage_text'] ,cut = True)

        result = search_start_end(pattern,sentence)

        sequence = sequence_label(sentence,list_words)

        line['passages'][i]['passage_text'] = "".join(list_words)

        if result is not None:
            # start to find answers
            try:
                start_pos = sequence[result[0]]
                #TODO: when result[1] is the length of the sequence ,it need to be reduced 1 to avoid arriving at the end 
                end_pos   = sequence[result[1]-1]
                result = (start_pos, end_pos)
            except:
                print("Error:start_pos:{},end_pos:{},sequence_len:{}".format(result[0],result[1],len(sequence)))
                result =None
                exit(0)

        line['passages'][i]['answer_point']  = result

        #find the result
        if result:
            per_count[index] =1

    if per_count[index] == 0:
        modify_list.append(index)

    if index %100 ==0:
        print(index)

    # append to the list   
    processed_data.append(line)

print("finded answers:{}".format(sum(per_count)))

print("data length:{}".format(len(processed_data)))

#now start to store the data we have
with  codecs.open(args.aim_path,'w','utf-8') as fp:
    for item in processed_data:
        fp.write(json.dumps(item,ensure_ascii = False) + '\n')
print("finished")
