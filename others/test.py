import seaborn as sns;

import model.model_add_aligned as model_add_aligned
# the commandline parameters
from  function.parameter import args
from function.reader import *

sns.set()
import matplotlib.pyplot as plt
logging.basicConfig(level=logging.INFO)


args.batch_size = 2

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

        
        feed = set_dict(trainModel,query_ls , passage_ls, answer_p_s, answer_p_e,passage_pos_ls,add_token_feature = args.add_token_feature)


        answer_one_hot = sess.run([trainModel.passage_logit_pro_start,trainModel.passage_logit_pro_end],feed_dict=feed) 


        toSee = [trainModel.p_We_q,trainModel.p_W_q,trainModel.seqmasked,trainModel.beforemasked,trainModel.selfAttenMask]
        
        p_We_q,p_W_q,masked,masked_before,selfAttenMask= sess.run(toSee,feed_dict=feed)
        ax = sns.heatmap(selfAttenMask.reshape( (args.batch_size,-1)),annot=True, cmap="YlGnBu")
        plt.show()

        # save summary
        exit(0)
    reader.reset()

print("finished")
