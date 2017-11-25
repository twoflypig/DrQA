import json
import codecs
import argparse
from ultize import *

parser = argparse.ArgumentParser(description='produce answer and passage vocab.')

parser.add_argument('--src_path',type= str ,default = "None",
                    help='the origin data path')
parser.add_argument('--vocab_path', type= str, default = "None",
                    help='the path of answer and passage vocab')
parser.add_argument('--add_answer', type= bool, default = True,
                    help='whether to add answer vocab')

Englist_TAG = 'EN'
Number_TAG = 'NUM'
Number_Number_Tag = 'NDOTN' 


with codecs.open("tag_vocab",'w','utf8') as fp:
    """
    build a tag dict
    """
    fp.write(Englist_TAG +'\n')
    fp.write(Number_TAG +'\n')
    fp.write(Number_Number_Tag +'\n')

jieba.load_userdict("tag_vocab")

def save(path,word_set):
    """
    save vocab to disk
    """
    with  codecs.open(path,'w','utf-8') as fp:

        # attention : here we add a </s> to indicate the end of sentence
        fp.write('</s>' +'\n')
        for item in word_set:
            fp.write(item + '\n' )
        fp.write('<unk>' +'\n') # in case there are no see words

args = parser.parse_args()

answer_words  =  []
passages_words=  []
query_words   =  []

with  codecs.open(args.src_path,"r","utf8") as fp:
    data = fp.readlines()

print("add_answer to vocab:{}".format(args.add_answer))

for index,item in  enumerate(data):

    line  = json.loads(item)
    #here vocab may exist some problems
    
    # adding query vocabs
    # query  =  replace_fuse( line['query'] ,Number_Number_Tag ,Number_TAG, Englist_TAG)
    query  =  cut_sentence( line['query']  ,cut =True)

    # replace numbers and englist 
    query =  check_nunber_en(query,tagnum = Number_TAG , tagen =Englist_TAG )

    query_words.extend(query)

    if args.add_answer :
        
        #print("adding answering...")
        # there is no need to process answer 

        answer_temp  =  process_answer(line['answer'])

        answer_1   =    answer_temp.split()

        answer_2   =    (convert_ch2num(answer_temp)).split()

        #print("index:{},answer:{}".format(index,answer_2) )

        answer_words.extend([answer_temp] + answer_1 + answer_2 )

    for i in range(len(line['passages'])):

        # passage_temp  = replace_fuse (line['passages'][i]['passage_text'] ,Number_Number_Tag ,Number_TAG, Englist_TAG)

        #print(passage_temp) 
        # firstly , delete all the english numbers and Arabic numerals
        # then adding character-level english numbers and Arabic numerals to vocab
       
        #list_words = process_line(line['passages'][i]['passage_text'] ,cut = True)

        list_words , _ = token_pos(line['passages'][i]['passage_text'] , use_pos = False)

        list_words = check_nunber_en(list_words,tagnum = Number_TAG , tagen =Englist_TAG)

        passages_words.extend(list_words)

    if index % 100 ==0:
        print(index)


answer_words =set(answer_words)

passages_words = set(passages_words)

query_words  =set(query_words)

passages_words |= answer_words

passages_words |= query_words

# words_list final answer words

save(args.vocab_path , passages_words )