import tensorflow as tf

class model(object):
    def __init__(self,config):
        self.config = config
        self.batch_size=  config.batch_size
        self.num_units  = config.num_units
        self.src_vocab_size  = config.src_vocab_size
        self.input_embedding_size = config.input_embedding_size
        self.max_gradient_norm = 1
        self.learning_rate = 0.001

    def MultiBiRNN(self,inputs,hidden_units,num_layers,sequence_length,dropout_output,name):

        output = inputs

        for i in range(num_layers):

            with  tf.variable_scope( name +str(i), reuse=False) as scope:
                
                if dropout_output < 1:
                    output = tf.nn.dropout(output,dropout_output) 

                forward_cell = tf.contrib.rnn.LSTMCell(hidden_units)
                backward_cell = tf.contrib.rnn.LSTMCell(hidden_units)

                output, _states = tf.nn.bidirectional_dynamic_rnn(forward_cell, backward_cell, output,
                                                sequence_length=sequence_length, time_major=True,dtype = tf.float32)
                output =  tf.concat(output, -1)
            print("{} layer {} : keep_pro:{}".format(name,i,dropout_output))

        if dropout_output < 1:
            output = tf.nn.dropout(output,dropout_output) 

        return output
    def SeqAtten(self,martrix_passage,matrix_query,martrix_passage_length,matrix_query_length,name):
        """
            func: Given sequences X and Y, match sequence Y to each element in X.
            input martrix_passage: length1,batch_size,feature
                  matrix_query   : length2,batch_size,feature
                  martrix_passage_length : batch_size,
                  matrix_query_length :batch_size
            return aligned_question_embeding: batch_size,length1,feature
        """
        with  tf.variable_scope( name , reuse=False) as scope:
            # TODO: Here we used reused needed to be valided
            with tf.variable_scope("aligned_dense") :
               alph_passage = tf.layers.dense(martrix_passage,martrix_passage.get_shape().as_list()[2],kernel_initializer =tf.random_uniform_initializer(-0.5, 0.5, seed=12),activation=tf.nn.relu)#,name='aligned_dense',reuse=None)
      
            with tf.variable_scope("aligned_dense", reuse=True):
               # shape of alph_question is [?,20,1]
               alph_question = tf.layers.dense(matrix_query ,matrix_query.get_shape().as_list()[2],kernel_initializer =tf.random_uniform_initializer(-0.5, 0.5, seed=12),activation=tf.nn.relu)#,name='aligned_dense',reuse=True)
       

            # result:[batch_size,? ,?] the result of matmul between [20,?,1] and [20,1,?]
            # so dimension = 1 means passage pi , and simension =2 means question qi
            batch_martix = tf.matmul( tf.transpose(alph_passage,[1,0,2]) , tf.transpose(alph_question,[1,2,0]) )

            # input query_sequence_length : [batch_size,]
            # input passage_sequence_length : [batch_size,]
            # the result of sequence_mask shape: batch,passage_length,query_length
            # returned masked shape [batch,passage_length,query_length]
            sequence_mask=  tf.tile(tf.expand_dims(tf.sequence_mask(matrix_query_length),axis=1),[1,tf.reduce_max( tf.reshape(martrix_passage_length,(-1,1))),1])

            sequence_ones = tf.ones_like(sequence_mask,dtype=tf.float32)

            sequence_infi = sequence_ones * (-float('inf'))

            masked = tf.where(sequence_mask,batch_martix,sequence_infi)
            # apply softmax ,dim default is -1 ,operates on the last dimension the batch_martix
            #shape is [batch_size,?,?]
            batch_softmax_martix = tf.nn.softmax(masked)
            #[20,?,200] =  [20,?,?] x [20,?,200]
            aligned_question_embeding=  tf.matmul(batch_softmax_martix, tf.transpose(matrix_query,[1,0,2]) )    
            print("building SeqAtten:{}".format(name))
            return aligned_question_embeding ,batch_softmax_martix,masked
    def bilineaAtten(self,matrix_passage,query,name):

        with  tf.variable_scope( name , reuse=False) as scope:
            bb_W =  tf.Variable(tf.truncated_normal([matrix_passage.shape.as_list()[2],query.shape.as_list()[0]], stddev=0.1))
            
            bb_W = tf.tile( tf.expand_dims(bb_W,dim=0),[self.batch_size,1,1])
            bb_W_q = tf.matmul(bb_W, tf.expand_dims(tf.transpose(query,[1,0]),dim=-1))
            bb_p = tf.transpose(matrix_passage,[1,0,2])
            bb_pwq = tf.matmul(bb_p,bb_W_q)
            p_W_q = tf.squeeze(bb_pwq,axis=2)

            tf.summary.histogram("bilineaAtten"+name,bb_W)
            print("building bilineaAtten:{}".format(name))
            return p_W_q
    def  SelfAtten(self,matrix,length,name):
        """
            func: Given sequences X , match attention to self.
            input matrix: length1,batch_size,feature
                  length   : batch_size,
 
            return aligned_question_embeding: feature,batch_size
        """
        with tf.name_scope(name) as scope:

            W = tf.Variable(tf.truncated_normal([matrix.get_shape().as_list()[2],1], stddev=0.1))

            tile_w = tf.tile(tf.expand_dims(W,axis = 0),[self.batch_size,1,1])

            weight_martix = tf.matmul(  tf.transpose(matrix,[1,0,2]),tile_w)

            sequence_masked  =  tf.expand_dims(tf.sequence_mask(length),axis=2)
            sequence_ones_1 = tf.ones_like(sequence_masked,dtype=tf.float32)
            sequence_infi_1 = sequence_ones_1 * (-float('inf'))

            weight_martix = tf.nn.softmax(tf.where(sequence_masked,weight_martix,sequence_infi_1),dim=1)

            final_query = tf.reduce_sum( tf.transpose(matrix,[0,2,1]) *  tf.transpose(weight_martix,[1,2,0]) ,axis =0)

            tf.summary.histogram("SelfAttenW"+name,W)

            print("building SelfAtten:{}".format(name))
        return final_query,weight_martix
    def build_model(self):
        # passage
        # as time-major
        with tf.variable_scope("input") as scope:
            # time major
            self.passage_inputs = tf.placeholder(shape=(None,self.batch_size), dtype=tf.int32, name='passage_inputs')
            self.passage_sequence_length = tf.placeholder(shape=([self.batch_size]), dtype=tf.int32, name='passage_length')
            # query
            self.query_inputs = tf.placeholder(shape=(None,self.batch_size), dtype=tf.int32, name='query_inputs')
            self.query_sequence_length = tf.placeholder(shape=([self.batch_size]), dtype=tf.int32, name='query_length')

            # binary_vector size is [max_length,batch_size]
            self.binary_inputs = tf.placeholder(shape=(None,self.batch_size), dtype=tf.float32, name='binary_vector')
            # answer_pi
            self.passage_start_pos =  tf.placeholder(shape=([self.batch_size]), dtype=tf.int32, name='passage_start_pos')
            self.passage_end_pos =  tf.placeholder(shape=([self.batch_size]), dtype=tf.int32, name='passage_end_pos')
            self.passage_logit_pro_start =tf.one_hot(self.passage_start_pos, depth= tf.reduce_max(self.passage_sequence_length) )     #tf.placeholder(shape=(1, passage_max), dtype=tf.int32, name='self.passage_logit_pro_start')
            self.passage_logit_pro_end = tf.one_hot(self.passage_end_pos,    depth= tf.reduce_max(self.passage_sequence_length) )  #tf.placeholder(shape=(1, passage_max), dtype=tf.int32, name='self.passage_logit_pro_end')
             #input is (nums,batch_sizes) result should be (?, 20, 401) (nums,batch_size,deeps)

            self.pos_passages_inputs = tf.placeholder(shape=(None,self.batch_size), dtype=tf.int32, name='pos_passages_vector')
            #input is (nums,batch_sizes) result should be (?, 20, 401) (nums,batch_size,deeps)
            passages_pos_vectors = tf.one_hot(tf.transpose(self.pos_passages_inputs,[1,0]),depth = self.config.pos_vocab_size,axis = -1)
            
            if self.config.add_token_feature:
                 print("passage_pos_vecots shape:{}".format(passages_pos_vectors.shape))    
            # embedding
            if self.config.use_pretrain_vector:

                # load pre-trained vector
                print("using pre-trained vector")
                self.embedding_placeholder = tf.placeholder(tf.float32, [self.config.pre_trained_embedding_length, self.input_embedding_size])
                embeddings = tf.Variable(tf.constant(0.0, shape=[self.config.pre_trained_embedding_length , self.input_embedding_size]),
                                    trainable=True, name="W")
                # embedding intial assgin op
                self.embedding_init = embeddings.assign(self.embedding_placeholder)
            else:
                print("using vocab vectos training with models")
                embeddings = tf.get_variable('W',[self.config.pre_trained_embedding_length , self.input_embedding_size],
                                                       initializer=tf.random_uniform_initializer(-0.2, 0.2, seed=123),
                                                        dtype=tf.float32)
            if self.config.is_training is False:
                print("Inference:contact unsee word embedding")
                # Unseembeddings = tf.get_variable('UnseeEmbedding',[ self.config.src_vocab_size - self.config.pre_trained_embedding_length , self.input_embedding_size],
                #                                        initializer=tf.random_uniform_initializer(-0.4, 0.4, seed=21),
                #                                         dtype=tf.float32,trainable= False)
                Unseembeddings = tf.Variable(tf.truncated_normal([ self.config.src_vocab_size - self.config.pre_trained_embedding_length, self.input_embedding_size],
                                                              stddev=0.1),
                                                        dtype=tf.float32,trainable= False)
                embeddings = tf.concat([embeddings,Unseembeddings],axis=0)

            passage_inputs_embedded = tf.nn.embedding_lookup(embeddings, self.passage_inputs)
            query_inputs_embedded = tf.nn.embedding_lookup(embeddings, self.query_inputs)

            # set global_step
            self.global_step = tf.Variable(0, trainable=False)


        aligned_question_embeding,self.seqmasked,self.beforemasked= self.SeqAtten(passage_inputs_embedded,query_inputs_embedded,self.passage_sequence_length,
                                            self.query_sequence_length,"q_p_alligned")


        with tf.name_scope("passage_rnn") :

            fuse_passage_encoding = tf.concat(  [passage_inputs_embedded ,tf.transpose(aligned_question_embeding ,[1,0,2]) ],axis =2 )
            
            # adding binary feature
            fuse_passage_encoding = tf.concat(  [fuse_passage_encoding ,tf.expand_dims(self.binary_inputs,axis=-1) ],axis =2 )
            
            if self.config.add_token_feature is True:
                # adding token feature
                print("Adding pos feature")
                fuse_passage_encoding = tf.concat(  [fuse_passage_encoding , tf.transpose(passages_pos_vectors,[1,0,2]) ],axis =2 )

            print("after adding pos vector, shape is:{}".format(fuse_passage_encoding.shape))



            passage_outputs =  self.MultiBiRNN(inputs=fuse_passage_encoding,
                                            hidden_units =self.num_units,
                                            num_layers = self.config.num_layer,
                                            sequence_length = self.passage_sequence_length,
                                            dropout_output =self.config.keep_pro ,
                                            name= "passage")

            

        with tf.name_scope("question_rnn") as scope:

            question_outputs =  self.MultiBiRNN(inputs=query_inputs_embedded,
                                hidden_units =self.num_units,
                                num_layers = self.config.num_layer,
                                sequence_length = self.query_sequence_length,
                                dropout_output =self.config.keep_pro ,
                                name= "query")
            final_query,self.selfAttenMask = self.SelfAtten(question_outputs,self.query_sequence_length,name="self-atten")

        with tf.name_scope('g') as scope:


                self.p_W_q =  self.bilineaAtten(passage_outputs,final_query,"pre_start")


                self.p_We_q =  self.bilineaAtten(passage_outputs,final_query,"pre_end")

        print("Training state :{}".format(self.config.is_training))
        if self.config.is_training is False :
            # I need to see probalities:
            self.end_pro = tf.exp(self.p_We_q)
            self.start_pro = tf.exp(self.p_W_q)
            print("In inference process")
            return
        with tf.name_scope("compute_loss") as scope:
            # start point
            

            pre_q_e_loss = tf.nn.softmax_cross_entropy_with_logits(labels= self.passage_logit_pro_end, logits=self.p_We_q)
            print("pre_q_e_loss shape:{}".format(pre_q_e_loss.get_shape()))
            self.cross_entropy_end = tf.reduce_mean(pre_q_e_loss)
            print("After reduce_mean loss shape:{}".format(self.cross_entropy_end.get_shape()))
            # end point
            pre_q_s_loss = tf.nn.softmax_cross_entropy_with_logits(labels= self.passage_logit_pro_start,logits=self.p_W_q)
            self.cross_entropy_start = tf.reduce_mean(pre_q_s_loss)


        with tf.name_scope("train_op") as scope:   

            parameters = tf.trainable_variables()
            gradients = tf.gradients(self.cross_entropy_start +self.cross_entropy_end , parameters)
            clipped_gradients, gradient_norm = tf.clip_by_global_norm(gradients, self.max_gradient_norm)

            """add train op"""
            optimizer = tf.train.AdamOptimizer( self.learning_rate)
            # Attention: here self.global_step will increment by one after the variables have been updated.
            self.train_op = optimizer.apply_gradients(zip(clipped_gradients, parameters),global_step= self.global_step)


        # record weight change 

        tf.summary.scalar("cross_entropy_start", self.cross_entropy_start)
        tf.summary.scalar("cross_entropy_end", self.cross_entropy_end)
        tf.summary.scalar("sum_loss", self.cross_entropy_start +self.cross_entropy_end )
        #tf.summary.scalar("gradient_norm", gradient_norm)



        #tf.summary.histogram("embeddings",embeddings)

        self.summary_op = tf.summary.merge_all()
