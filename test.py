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

logging.basicConfig(level=logging.INFO)


args.batch_size = 12

# Read cha_vectors.bin
if args.use_pretrain_vector is False:
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
reader  = Reader(args,vocab)
args.src_vocab_size = vocab_size
args.input_embedding_size = embedding_dim
args.pos_vocab_size =  len(reader.pos_vocab)  # size of vocab


trainModel = model_add_aligned.model(args)
trainModel.build_model()

# para_config = tf.ConfigProto(
#                 inter_op_parallelism_threads = 2,
#                 intra_op_parallelism_threads = 10)

sess = tf.Session()#config=para_config)
saver = tf.train.Saver()
ckpt_state = tf.train.get_checkpoint_state(args.restore_path)



if ckpt_state == None:
    print("Cant't load model,starting initial")
    exit(0)
    # load embedding
    if args.use_pretrain_vector:
        sess.run(trainModel.embedding_init, feed_dict={trainModel.embedding_placeholder: embd})
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

for m_epoch in range(args.epoch):

    per_loss_start = 0
    
    per_loss_end = 0
    
    for step in  range(reader.num_examples // args.batch_size):

        query_ls , passage_ls, answer_ls, answer_p_s, answer_p_e,passage_pos_ls,ori_pas,ori_query = reader.get_batch(True)

       # logging.info("query_ls:{},passage_ls:{},answer_ls:{},answer_p_s:{},answer_p_e:{},passage_pos_ls:{}".format(

       #        query_ls,passage_ls,answer_ls, answer_p_s,answer_p_e,passage_pos_ls))
        
        feed = set_dict(trainModel,query_ls , passage_ls, answer_p_s, answer_p_e,passage_pos_ls,add_token_feature = args.add_token_feature)
        
       # logging.info("feed.binary:{}".format(feed[trainModel.binary_inputs]))

        answer_one_hot = sess.run([trainModel.passage_logit_pro_start,trainModel.passage_logit_pro_end],feed_dict=feed) 

        #logging.info("answer_one_hot:{},max_length:{}".format(answer_one_hot,max(feed[trainModel.passage_sequence_length])))

        toSee = [trainModel.p_We_q,trainModel.p_W_q,trainModel.seqmasked,trainModel.beforemasked,trainModel.selfAttenMask]
        
        p_We_q,p_W_q,masked,masked_before,selfAttenMask= sess.run(toSee,feed_dict=feed)

        #debug_fp.write("".joint)
        

        #print()
        print("p_We_q:{},p_W_q:{},masked shape:{},maksed value:{}".format(p_We_q,p_W_q,masked.shape,masked))
        # save summary
        exit(0)
    reader.reset()

print("finished")
