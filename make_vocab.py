"""
This file is used to create a vocab file

"""
import codecs
import argparse
from ultize import *
from collections import Counter

parser = argparse.ArgumentParser(description='produce answer and passage vocab.')

parser.add_argument('--src_path',type= str ,default = "None",
                    help='the origin data path')
parser.add_argument('--vocab_path', type= str, default = "None",
                    help='the path of answer and passage vocab')
parser.add_argument('--add_answer', type=lambda s: s.lower() in ['true', 't', 'yes', '1'] ,default= True,
                    help='whether to add answer vocab')
parser.add_argument('--add_endtag', type=lambda s: s.lower() in ['true', 't', 'yes', '1'] ,default= True,
                    help='whether to add </s> and <unk>,used in inference')
Englist_TAG = 'EN'
Number_TAG = 'NUM'


def save(path,word_set,add_tag):
    """
    save vocab to disk
    """
    with  codecs.open(path,'w','utf-8') as fp:

        # attention : here we add a </s> to indicate the end of sentence
        for item in word_set:
            fp.write(item[0] + '\n' )  # the input word_set is a tuple (word,counts)
        if add_tag:
            print("Adding </s> and <unk> to the vocab")
            fp.write('</s>' + '\n')
            fp.write('<unk>' +'\n') # in case there are no see words
        else:
            print("Not adding </s> and <unk> to the vocab")


def get_vocab_list(path,add_answer ):
    answer_words = []
    passages_words = []
    query_words = []
    with  codecs.open(path,"r","utf8") as fp:
        data = fp.readlines()

    print("add_answer to vocab:{}".format(add_answer))

    for index,item in  enumerate(data):

        line  = json.loads(item)
        #here vocab may exist some problems

        # adding query vocabs
        # query  =  replace_fuse( line['query'] ,Number_Number_Tag ,Number_TAG, Englist_TAG)
        query  =  cut_sentence( line['query']  ,cut =True)

        # replace numbers and englist
        query =  check_nunber_en(query,tagnum = Number_TAG , tagen =Englist_TAG )

        query_words.extend(query)

        if add_answer :

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

    passages_words.extend(answer_words)

    passages_words.extend(query_words)

    passages_count = Counter(passages_words)

    words_result = passages_count.most_common()

    return words_result

args = parser.parse_args()

words_result = get_vocab_list(args.src_path,args.add_answer)

print("vocab size:{}".format(len(words_result)))

save(args.vocab_path , words_result, args.add_endtag)