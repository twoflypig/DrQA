#coding=utf-8
import codecs
import numpy as np
import json
import re
import jieba
import jieba.posseg as pseg
"""
process needed
"""
re_delete_space = re.compile(r'\u3000| |\.{2,10}|\xa0|\u2002|\u2003|\u202f')
#This is a regex that only keep number
rule = re.compile(r"[^a-zA-Z0-9，。！？,.?!<<>>《》()（）\"%+-×/\{}|～ ~u4e00-\u9fa5]")

# here for vocab making 
# converting numbers and Arabic numerals to tags
rule_num = re.compile(r"[0-9]+") 
rule_englist = re.compile(r"[a-zA-Z]+") 
rule_numdotnum =re.compile(r"\d+\.\d+")


number_dict={'三十':'30','九':'9','八':'8','七':'7','六':'6','五':'5','四':'4','三':'3','二':'2','一':'1','零':'0'}
reverse_number_dict = {v:k for k, v in number_dict.items() }

def delete_replicate(oldstring):
    """
    the oldstring must be origin sentence 
    delete replicate charcters
    """
    try:
        if oldstring != "":
            newstring = oldstring[0]
            for char in oldstring[1:]:
                if char != newstring[-1]:
                    newstring += char
            return newstring
        else:
            return oldstring
    except:
        print(oldstring=="")
        print(oldstring)
        exit(0)
        

def strQ2B(ustring):
    """全角转半角"""
    rstring = ""
    for uchar in ustring:
        inside_code=ord(uchar)
        if inside_code == 12288:                              #全角空格直接转换
            inside_code = 32
        elif (inside_code >= 65281 and inside_code <= 65374): #全角字符（除空格）根据关系转化
            inside_code -= 65248

        rstring += chr(inside_code)
    return rstring


def convert_num2ch(string):
    """
    convert numbers in Alabo  to numbers in chinese
    """
    temp_str=[]
    for item in string:
        temp_str.append( reverse_number_dict.get(item,item))
    return "".join(temp_str)

def convert_ch2num(string):
    temp_str=[]
    for item in string:
        temp_str.append( number_dict.get(item,item))
    return "".join(temp_str)

def replace_numdotnum_withtag(sentence,tag):
    
    # input tag must be a string
    # replacing strings like 3.2
    # used in make_vocab.py
    line = re.sub(rule_numdotnum,tag,sentence)

    return line 

def replace_num_withtag(sentence,tag):
    
    # input tag must be a string
    # replacing strings like 32
    # used in make_vocab.py
    line = re.sub(rule_num,tag,sentence)

    return line 

def replace_englist_withtag(sentence,tag):
    
    # input tag must be a string
    # replacing strings like  apple 
    # used in make_vocab.py
    line = re.sub(rule_englist,tag,sentence)

    return line 

def replace_fuse(sentence,tag_nn,tagnum,tagen):
    # performing vocab replacing all 
    line = replace_englist_withtag(sentence , tagen)
    line = replace_numdotnum_withtag(line,tag_nn)
    line = replace_num_withtag(line , tagnum)

    return line 

def check_nunber_en(words_ls , tagnum,tagen):

    result = []
    for item in words_ls:
        if re.match(rule_num,item) :
            result.append(tagnum)
        elif re.match(rule_englist,item):
            result.append(tagen)
        else:
            result.append(item)
    return result

def process_line(line,cut = True):
    
    """
    if cut is Ture:
        return a list of words,which are processed by regex
    else:
        just return the word processed by regex
    """

    # deleting replicate characters
    if line is None:
        return None
    line = strQ2B(line)

    str0 = delete_replicate(line)

    # deleting special space  characters , spaces are also removed for convient
    str1 = re.sub(re_delete_space,"",str0)
    # remove others special characters,only allowed chacters are keeped
    str2 = re.sub(rule,"",str1)

    if cut is True:
        # if there are spaces ? how to process ? not allowed to be cut!
        cut_down = jieba.lcut( str2,cut_all = False)

        return cut_down
    else:

        return str2

def process_answer(line):
    """
    input must be a string ,remove some unrelated flags
    """
    temp_line  = re.sub(re_delete_space," ",strQ2B(line))
    # make sure we can search for answers with space
    line_split = temp_line.split()
    if len(line_split) >1:
        temp_line = "".join(line_split)
    return process_replace_brackets(temp_line)



def process_replace_brackets(line):
    return line.replace('(','\(').replace(')','\)')

def cut_sentence(sentence,cut = False):
    if cut:
        return jieba.lcut(sentence,cut_all = False)
    else:
        return sentence

def token_pos(sentence , use_pos = True):
    """
    if use_pos is True: return pos , else return empty pos
    input: a strng
    return tokenize ,part of speech ,
    attention: the result of tokenize may not same as the result of lcut 
    """
    if use_pos :
        result =  pseg.cut(sentence)

        words_ls = []
        pos_ls = []

        for word, pp in result:
            words_ls.append(word)
            pos_ls.append(pp)

        return words_ls,pos_ls
    else:
        words_ls  = cut_sentence(sentence ,cut = True)

        # attention: here end should an intance of pos_vocab

        return words_ls, ['end']*len(words_ls)


def loadWord2Vec(filename):
    # filename: *.bin  is the result of word2vec
    vocab = []
    cnt = 0
    fr = codecs.open(filename,'r','utf-8')
    line = fr.readline().strip()
    #print line
    word_dim = int(line.split(' ')[1])
    vocab_size = int(line.split(' ')[0])
    buffer_vector=  np.zeros( (vocab_size,word_dim))
    #vocab.append("<unk>")
    print("vector shape is:{}".format(buffer_vector.shape))
    for index,line in enumerate(fr) :
        row = line.strip().split(' ')
        # if len(row) == word_dim +1:
        vocab.append(row[0])
        value_line = np.array( [float(x) for x in row[1:] ] )
        buffer_vector[index,:] =  value_line
        # else:
        #     # TODO: some lines are ignored!
        #     vocab.append(line[0])
        #     value_line = np.array( [float(x) for x in row[0:] ] )
        #     buffer_vector[index,:] =  value_line
    print("loaded word2vec")
    fr.close()
    return vocab,buffer_vector

def loadvocab(filename):
    vocab= []
    with codecs.open(filename,'r','utf-8') as fp:
        for line in fp:
            vocab.append(line.strip())
    return vocab

def load_pos_vocab(filename):
    # load pos vocab
    vocab = []
    with codecs.open(filename,'r','utf-8') as fp:
        for line in fp:
            vocab.append(line.strip())
    return vocab

def id2word(sentence, id_vocab):
    """
    sentence: input a id list ,witch is designed for character level
    vocab: a dict, key is character,value is index
    return a string
    """
    ids = []
    for p_id in sentence:
        ids.append( id_vocab.get(p_id,"unk") )
    return " ".join(ids)
def batch2id(pair,vocab):
    """
    pair: the list of  q,a,d
    vocab:vocab
    """
    result = [ word2id(item,vocab) for item in pair]

    return result
def batchlize(inputs, max_sequence_length=None):
    """
    convert list to numpy martix
    """
    if  max_sequence_length:
        sequence_lengths=[]
        for seq in inputs:
            if len(seq)>=max_sequence_length:
                sequence_lengths.append(max_sequence_length)
            else:
                sequence_lengths.append(len(seq))
    else:
        sequence_lengths = [len(seq) for seq in inputs]

    batch_size = len(inputs)

    # print("batch_size:{}".format(batch_size))

    max_sequence_length = max(sequence_lengths)

    # print("max_sequence_length:{}".format(max_sequence_length))

    inputs_batch_major = np.zeros(shape=[batch_size, max_sequence_length], dtype=np.int32) # == PAD

    for i, seq in enumerate(inputs):
        for j, element in enumerate(seq):
            if j >= max_sequence_length:
                break
            else:
                # because inputs are None
                try:
                    inputs_batch_major[i, j] = element
                except:
                    print("error none locate at {}".format(i))
                    exit(0)

    # [batch_size, max_time] -> [max_time, batch_size]
    inputs_time_major = inputs_batch_major.swapaxes(0, 1)

    return inputs_time_major, sequence_lengths
def check_exis_question(passage_ls,query_ls):
    """
    passage_ls:a id list
    query_ls : a id list
    func: check whether the id in passage appear in the correspondding questions,return a martix including 1 or 0 ,1 means ture, 0 means not find ,for each word in passage
    then return a numpy martix ,the same dimension as passage inputs
    """
    binary_ls = []
    for index, passage in  enumerate(passage_ls):
        binary_per = []
        for i , word in enumerate(passage):
            if word in query_ls[index]:
                binary_per.append(1)
            else:
                binary_per.append(0)
        binary_ls.append(binary_per)
    return batchlize(binary_ls)

# feed the return numpy array to dict
def set_dict(model, query_ls , passage_ls, answer_p_s, answer_p_e,passage_pos_ls,add_token_feature = False):
    
    passage_batch , passage_length, query_batch,query_length,binary_batch ,passage_pos_batch = \
             get_numpys(query_ls , passage_ls, passage_pos_ls)

    feed={
      model.passage_inputs:passage_batch,
      model.passage_sequence_length:passage_length,
      model.query_inputs: query_batch,
      model.query_sequence_length:query_length,
      model.passage_start_pos:answer_p_s,
      model.passage_end_pos:answer_p_e,
      model.binary_inputs:binary_batch,
      model.pos_passages_inputs:passage_pos_batch
     }
    return feed
# get numpys forms 
def get_numpys(query_ls , passage_ls,passage_pos_ls,add_token_feature = False):
    """
    inputs:all inputs must be list
    convert inputs list of ids to numpy martix,and check binary feature ,which are also converted to numpy martix
    """
    #print("starting...")
    query_batch , query_length = batchlize(query_ls)
    #print("After query_ls")
    passage_batch , passage_length = batchlize(passage_ls)
    #print("After passage_ls")
    binary_batch ,_ = check_exis_question(passage_ls,query_ls)
    #print("After binary_batch")
    # print("query_ls:{}".format(query_ls))

    # print("query_pos_ls:{}".format(query_pos_ls))

    # print("len of query pos :{}".format( len(query_pos_ls)))

    # print("passage_ls:{}".format(passage_ls))

    # print("passage_pos_ls:{}".format(passage_pos_ls))

    # print("len of passage pos :{}".format( len(passage_pos_ls)))
    passage_pos_batch, _ = batchlize(passage_pos_ls)
    #print("After passage_pos")
    # add_token_feature is False , set passage_pos_batch to be zeros
    if add_token_feature is False:
        passage_pos_batch = np.zeros_like(passage_pos_batch)

    return  passage_batch , passage_length, query_batch,query_length,binary_batch ,passage_pos_batch