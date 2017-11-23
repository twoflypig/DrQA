import json
import codecs
import argparse
from ultize import *

parser = argparse.ArgumentParser(description='produce answer and passage vocab.')

parser.add_argument('--src_path',type= str ,default = "None",
                    help='the origin data path')
parser.add_argument('--answer_path', type= str, default = "None",
                    help='the path of answer vocab')
parser.add_argument('--vocab_path', type= str, default = "None",
                    help='the path of answer and passage vocab')
parser.add_argument('--add_answer', type= bool, default = True,
                    help='whether to add answer vocab')


args = parser.parse_args()

answer_words= []
passages_words=[]


with  codecs.open(args.src_path,"r","utf8") as fp:
    data = fp.readlines()


for index,item in  enumerate(data):

    line  = json.loads(item)
    #here vocab may exist some problems
    if args.add_answer:
        
        answer_1 = process_answer(line['answer']).split()

        answer_2 = process_answer(convert_num2ch(line['answer'])).split()

        answer_words.extend(answer_1 + answer_2)

    for i in range(len(line['passages'])):
       
        #list_words = process_line(line['passages'][i]['passage_text'] ,cut = True)

        list_words , _ = token_pos(line['passages'][i]['passage_text'])

        passages_words.extend(list_words)

    if index % 100 ==0:
        print(index)

answer_words =set(answer_words)

passages_words = set(passages_words)

# cut words
words_list = []

for item in answer_words:
	words_list.extend(process_line(item,cut=True))
else:
    print("not adding answers")
print("words set before:{}".format(len(words_list)))
words_list=set(words_list)
print("words set after:{}".format(len(words_list)))

def save(path,word_set):
    with  codecs.open(path,'w','utf-8') as fp:

        # attention : here we add a </s> to indicate the end of sentence
        fp.write('</s>' +'\n')
        for item in word_set:
            fp.write(item + '\n' )

passages_words |= words_list

# words_list final answer words

if args.add_answer:
    save(args.answer_path, words_list )
save(args.vocab_path , passages_words )