import argparse
import tensorflow as tf
from model import model
from reader import *
from ultize import *
import numpy as np
from collections import Counter
import model.model_add_aligned as model_add_aligned
import time
import logging
parser = argparse.ArgumentParser(description='parameters.')
parser.add_argument('--batch_size',type= int ,default = 1,
                    help='the origin data path')
parser.add_argument('--num_units', type= int, default = 200,
                    help='the path of processed data ')
parser.add_argument('--is_training', type= bool, default = False, # attention : no matter what we pass ,result always is True
                    help='Ture means inference')
parser.add_argument('--restore_path', type= str, default = "../modelRestor/word-level/",
                    help='the path of retore path ')
parser.add_argument('--src_vocab_size', type= int, default = 17689,
                    help='the size of vocab size ')
parser.add_argument('--vocab_path', type= str, default = None,
                    help='the size of vocab size ')
parser.add_argument('--input_embedding_size', type= int, default = 200,
                    help='the size of embedding size ')
parser.add_argument('--data_path', type= str, default = "../input/data",
                    help='the path of data')
parser.add_argument('--vector_path', type= str, default = "../cha_vectors.bin",
                    help='the path of vector and vocab')
parser.add_argument('--result_path', type= str, default = "../output/result",
                    help='the path of result')
parser.add_argument('--test', type= str, default = "inference",
                    help='whether to check run ok')
parser.add_argument('--num_layer', type= int, default =3,
                    help='layers in biRNN')
parser.add_argument('--epoch', type= int, default =10,
                    help='the training epochs')
parser.add_argument('--pos_vocab_path', type= str, default ="pos_vocab",
                    help='the pos vocab')
parser.add_argument('--pos_vocab_size', type= int, default = 30,
                    help='the pos vocab size')
# this need to be set in train.py
parser.add_argument('--add_token_feature', type= bool, default = False,
                    help='add_token_feature to be Ture of False')

args = parser.parse_args()

# Read cha_vectors.bin
if args.vocab_path  is not None:
    vocab = loadvocab(args.vocab_path)
    vocab_size = len(vocab)
    embedding_dim = args.input_embedding_size 
    print("load vocab")
else:    
    vocab,embd = loadWord2Vec(args.vector_path)
    vocab_size = len(vocab)
    embedding_dim = len(embd[0])  
    print("load vector")  

vocab_index = range(vocab_size)
vocab = dict(zip(vocab,vocab_index)) # vocab
id_vocab = {v:k for k, v in vocab.items() }

# Define reader
reader  = infe_reader(args,vocab)
args.src_vocab_size = vocab_size
args.input_embedding_size = embedding_dim
args.pos_vocab_size =  len(reader.pos_vocab)  # size of vocab

evaluate_model = model_add_aligned.model(args)
evaluate_model.build_model()

# para_config = tf.ConfigProto(
#                 inter_op_parallelism_threads = 2,
#                 intra_op_parallelism_threads = 10)

sess = tf.Session()#config=para_config)
saver = tf.train.Saver()
ckpt_state = tf.train.get_checkpoint_state(args.restore_path)


if ckpt_state == None:
    print("Cant't load model")
    exit(0)
else:
    try:
        saver.restore(sess, ckpt_state.model_checkpoint_path)
        print("restor model successed")
    except:
        print("loading error.")
        exit(0)

# start to inference
print("model path:".format(args.restore_path))


reader.reset()


result_list= []
unkown_counts = 0
result_fp =  codecs.open(args.result_path,'w','utf-8')
start_time = time.time()
if args.test == "test":
    print("infernece testing...")
    Lenght =  10
else:
    Lenght = reader.length

for step in range(Lenght):

        print('step:'+ str(step))

        query_ls , passage_ls,query_id_ls,origin_passage,passage_pos_ls = reader.get_batch()

        print(len(origin_passage))

        passage_batch , passage_length, query_batch,query_length,binary_batch ,passage_pos_batch= \
                          get_numpys(query_ls , passage_ls,passage_pos_ls,args.add_token_feature)

        result_buffer= []

        for i in range(len(passage_ls)):
            feed={
              evaluate_model.passage_inputs:passage_batch[:,i].reshape((-1,1)),
              evaluate_model.passage_sequence_length:[passage_length[i]],
              evaluate_model.query_inputs: query_batch[:,i].reshape((-1,1)),
              evaluate_model.query_sequence_length:[query_length[i]],
              evaluate_model.binary_inputs:binary_batch[:,i].reshape((-1,1)),
              evaluate_model.pos_passages_inputs:passage_pos_batch[:,i].reshape((-1,1))
             }
            #print(feed[evaluate_model.pos_passages_inputs])
            #pre_s ,pre_e =sess.run([evaluate_model.start_pro,evaluate_model.end_pro],feed_dict=feed)
            #we use theunnormalized exponential and take argmax over all considered paragraph spans for our ﬁnal prediction
            pre_s ,pre_e =sess.run([evaluate_model.start_pro,evaluate_model.end_pro],feed_dict=feed)
            s_p = np.argmax(pre_s[0])
            e_p = np.argmax(pre_e[0])
            s_p_max = np.max(pre_s[0])
            e_p_max = np.max(pre_e[0])
            if s_p <e_p and s_p +5 >e_p:

                passage_split = origin_passage[i]
                #print(passage_split)
                #print(pre_s[0])
                buffer_answer = "".join(passage_split[s_p:e_p]) #id2word(passage_ls[i][s_p:e_p] ,id_vocab)
                
                #print(buffer_answer)
                result_buffer.append( (query_id_ls[0],buffer_answer,s_p,e_p,s_p_max*e_p_max))
                #print(id2word(query_ls[i],id_vocab) + '\t'+ id2word(passage_ls[i],id_vocab)+ buffer_answer +":" +str(s_p_max)+"\t"+ str(e_p_max))
                #result_buffer.append(buffer_answer)
                # result_fp.write(id2word(query_ls[i],id_vocab) + '\t'+ id2word(passage_ls[i],id_vocab)
                #            + buffer_answer +":" +str(s_p_max)+"\t"+ str(e_p_max) +'\n')
        if len(result_buffer):
            line = max(result_buffer,key = lambda item:item[3])
            print("pro:{},chosing:{}".format(line[3],line[1]))
            result_list.append( line)
        else:
            result_list.append( (query_id_ls[0],'None'))
            print("None")
            unkown_counts+=1
end_time =time.time()
print("spend:{}".format(end_time-start_time))
for line in result_list:
    result_fp.write( str(line[0])+'\t'+line[1]+'\n')
result_fp.close()
print("unknow:{}".format(unkown_counts))
