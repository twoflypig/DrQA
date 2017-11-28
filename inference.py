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

# the commandline parameters 
from  parameter import args

args.batch_size =1
args.is_training = False
args.test = "inference"


# Read cha_vectors.bin
if args.use_pretrain_vector  is  False:
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
            s_p_max = pre_s[0][s_p]#np.max(pre_s[0])
            e_p_max = pre_s[0][e_p]#np.max(pre_e[0])
            if s_p <e_p and s_p +5 >e_p :#and s_p!= len(origin_passage[i])-1 :

                
                passage_split = origin_passage[i]
                #print(passage_split)
                #print(pre_s[0])
                buffer_answer = "".join(passage_split[s_p:e_p]) 

                max_pro = s_p_max*e_p_max
                print("s_p:{},e_p:{},pro:{},query:{},answer:{},pasage len:{}".format(s_p,e_p,max_pro,id2word(query_ls[i],id_vocab),buffer_answer,len(passage_split)))
                #print(buffer_answer)
                result_buffer.append( (query_id_ls[0],buffer_answer,s_p,e_p,max_pro))
                #print(id2word(query_ls[i],id_vocab) + '\t'+ id2word(passage_ls[i],id_vocab)+ buffer_answer +":" +str(s_p_max)+"\t"+ str(e_p_max))
                #result_buffer.append(buffer_answer)
                # result_fp.write(id2word(query_ls[i],id_vocab) + '\t'+ id2word(passage_ls[i],id_vocab)
                #            + buffer_answer +":" +str(s_p_max)+"\t"+ str(e_p_max) +'\n')
        if len(result_buffer):
            line = max(result_buffer,key = lambda item:item[3])
            print("In integration pro:{},finally chosing:{}".format(line[4],line[1]))
            result_list.append( line)
        else:
            result_list.append( (query_id_ls[0],'None'))
            print("In integration pro:{},finally chosing:{}".format(0,"None"))
            unkown_counts+=1
end_time =time.time()
print("spend:{}".format(end_time-start_time))
for line in result_list:
    result_fp.write( str(line[0])+'\t'+line[1]+'\n')
result_fp.close()
print("unknow:{}".format(unkown_counts))
