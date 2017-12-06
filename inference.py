import argparse
import tensorflow as tf
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


# Read cha_vectors.bin or load vocab
if args.use_pretrain_vector  is  False:
    vocab = loadvocab(args.vocab_path)
    vocab_size = len(vocab)
    embedding_dim = args.input_embedding_size 
    print("load training vocab")
else:    
    vocab,embd = loadWord2Vec(args.vector_path)
    vocab_size = len(vocab)
    embedding_dim = len(embd[0])  
    print("load vector")

args.pre_trained_embedding_length = vocab_size

# load inference vocab
infer_vocab = loadvocab(args.infer_vocab_path)
infer_vocab_size = len(infer_vocab)
print("load inference vocab")
diff_vocab = get_diff_vocabs(vocab,infer_vocab)

# Adding inference vocab to the src vocab
vocab.extend(diff_vocab)
print("extend vocab size:{}".format(len(diff_vocab)))
vocab = dict(zip(vocab,range(len(vocab)))) # vocab
id_vocab = {v:k for k, v in vocab.items()}

# the embedding can be divided into two parts ,the known matrix and unknown matrix and the contact them



# Define reader
reader  = infer_reader(args,vocab)
args.src_vocab_size = len(vocab)
args.input_embedding_size = embedding_dim

args.pos_vocab_size =  len(reader.pos_vocab)  # size of vocab



evaluate_model = model_add_aligned.model(args)
evaluate_model.build_model()

# para_config = tf.ConfigProto(
#                 inter_op_parallelism_threads = 2,
#                 intra_op_parallelism_threads = 10)

sess = tf.Session()#config=para_config)

# Note: This will ignore the unsee variables when restoring the models
sess.run(tf.global_variables_initializer())
ckpt_state = tf.train.get_checkpoint_state(args.restore_path)
saver = tf.train.Saver(var_list=optimistic_restore_vars(ckpt_state.model_checkpoint_path) if ckpt_state else None)

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
if args.test_inference is True:
    Lenght = 10
    print("Infernece testing length:{}".format())
else:
    print("Starting inference")
    Lenght = reader.length

store_json_list= [] # result list to write

for step in range(Lenght):

        print('step:'+ str(step))

        query_ls , passage_ls,query_id_ls,origin_passage,passage_pos_ls = reader.get_batch()

        print(len(origin_passage))

        # passage_batch , passage_length, query_batch,query_length,binary_batch ,passage_pos_batch= \
        #                   get_numpys(query_ls , passage_ls,passage_pos_ls,args.add_token_feature)

        result_buffer= []
        store_json_dict= {} # now data
        for i in range(len(passage_ls)):

            passage_batch , passage_length, query_batch,query_length,binary_batch ,passage_pos_batch= \
            get_numpys( [query_ls[i]] , [passage_ls[i]],[passage_pos_ls[i]],args.add_token_feature)
            #print(passage_batch.shape)

            feed={
              evaluate_model.passage_inputs:passage_batch[:,0].reshape((-1,1)),
              evaluate_model.passage_sequence_length:[passage_length[0]],
              evaluate_model.query_inputs: query_batch[:,0].reshape((-1,1)),
              evaluate_model.query_sequence_length:[query_length[0]],
              evaluate_model.binary_inputs:binary_batch[:,0].reshape((-1,1)),
              evaluate_model.pos_passages_inputs:passage_pos_batch[:,0].reshape((-1,1))
             }
            #print(feed[evaluate_model.pos_passages_inputs])
            #pre_s ,pre_e =sess.run([evaluate_model.start_pro,evaluate_model.end_pro],feed_dict=feed)
            #we use theunnormalized exponential and take argmax over all considered paragraph spans for our ﬁnal prediction
            pre_s ,pre_e =sess.run([evaluate_model.start_pro,evaluate_model.end_pro],feed_dict=feed)
            s_p = np.argmax(pre_s[0])
            e_p = np.argmax(pre_e[0])
            s_p_max = pre_s[0][s_p]#np.max(pre_s[0])
            e_p_max = pre_s[0][e_p]#np.max(pre_e[0])
            if (s_p <e_p and s_p +5 >e_p) or args.show_self_define :#and s_p!= len(origin_passage[i])-1 :

                passage_split = origin_passage[i]

                buffer_answer = "".join(passage_split[s_p:e_p]) 

                max_pro = s_p_max*e_p_max
                print("s_p:{},e_p:{},pro:{},query:{},answer:{},pasage:{}".format(s_p,e_p,max_pro,id2word(query_ls[i],id_vocab),
                    buffer_answer,
                    passage_split if args.show_self_define else len(passage_split)))

                result_buffer.append( (query_id_ls[0],buffer_answer,s_p,e_p,max_pro))


        if len(result_buffer):
            line = max(result_buffer,key = lambda item:item[4])
            print("In integration pro:{},finally chosing:{}".format(line[4],line[1]))
            result_list.append( line)
        else:
            result_list.append( (query_id_ls[0],'None'))
            print("In integration pro:{},finally chosing:{}".format(0,"None"))
            unkown_counts+=1

        # store answer we produce
        store_json_dict['query_id']  = int(query_id_ls[0])
        store_json_dict['query']   =  id2word(query_ls[0],id_vocab)
        store_json_dict['answer_ls'] = make_answer_dict(result_buffer)

        store_json_list.append( json.dumps(store_json_dict,ensure_ascii = False))
        
end_time =time.time()
print("spend:{}".format(end_time-start_time))
for line in result_list:
    result_fp.write( str(line[0])+'\t'+line[1]+'\n')
result_fp.close()
print("unknow:{}".format(unkown_counts))

path = "../output/infer_answer.json"
print("writing answers to {}".format(path))
with codecs.open(path,'w','utf8') as fp:
    for line in store_json_list:
        fp.write(line +'\n')
print("writing finished")

