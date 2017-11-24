import gensim
import codecs
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='produce answer and passage vocab.')

parser.add_argument('--src_path',type= str ,default = "None",
                    help='the origin data path')
parser.add_argument('--answer_path', type= str, default = "None",
                    help='the path of answer vocab')
parser.add_argument('--vocab_path', type= str, default = "None",
                    help='the path of answer and passage vocab')
parser.add_argument('--add_answer', type= bool, default = True,
                    help='whether to add answer vocab')