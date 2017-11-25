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

parser.add_argument('--batch_size',type= int ,default = 20,
                    help='the origin data path')
parser.add_argument('--num_units', type= int, default = 200,
                    help='the path of processed data ')
parser.add_argument('--is_training', type=lambda s: s.lower() in ['true', 't', 'yes', '1'] ,default= False,
                    help='Ture means inference')
parser.add_argument('--restore_path', type= str, default = "../modelRestor/word-level/",
                    help='the path of retore path ')
parser.add_argument('--src_vocab_size', type= int, default = 17689,
                    help='the size of vocab size ')



parser.add_argument('--input_embedding_size', type= int, default = 200,
                    help='the size of embedding size ')
parser.add_argument('--data_path', type= str, default = "../input/data",
                    help='the path of data')
# check to use vocab or pre-train vector
parser.add_argument('--use_pretrain_vector',type=lambda s: s.lower() in ['true', 't', 'yes', '1'] ,default= False,
                    help='a switch to use pre-trained vector ')

parser.add_argument('--vocab_path', type= str, default = None,
                    help='the size of vocab size ')
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
# pos 
parser.add_argument('--pos_vocab_path', type= str, default ="pos_vocab",
                    help='the pos vocab')
parser.add_argument('--pos_vocab_size', type= int, default = 30,
                    help='the pos vocab size')
# this need to be set in inference.py
parser.add_argument('--add_token_feature', type=lambda s: s.lower() in ['true', 't', 'yes', '1'] ,default= False,
                    help='add_token_feature to be Ture of False')
# model version 
parser.add_argument('--version', action='version', version='%(prog)s 1.01')


args = parser.parse_args()


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
reader  = Reader(args,vocab)
args.src_vocab_size = vocab_size
args.input_embedding_size = embedding_dim
args.pos_vocab_size =  len(reader.pos_vocab)  # size of vocab

reader.reset()
for m_epoch in range(args.epoch):

    per_loss_start = 0
    
    per_loss_end = 0
    
    for step in  range(reader.num_examples // args.batch_size):

        query_ls , passage_ls, answer_ls, answer_p_s, answer_p_e,passage_pos_ls = reader.get_batch()
        
    reader.reset()

print("finished")
