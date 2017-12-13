import gensim
import codecs
import numpy as np
import argparse
from functions import save_vocab

parser = argparse.ArgumentParser(description='produce answer and passage vocab.')

parser.add_argument('--src_path', type=str, default="None",
                    help='the origin data path')
parser.add_argument('--vocab_path', type=str, default="None",
                    help='the path of vocab')
parser.add_argument('--aim_path', type=str, default="None",
                    help='specify where the vector to be saved')
args = parser.parse_args()

def vector2str(vector):
    """
    convert a numpy array to a list
    :param vector:
    :return: a str list
    """
    return [str(x) for x in vector.tolist()]

def save_vector(path,vocab,vector,dimension):
    """
    save vector to disk
    :param path: a string,
    :param vocab: a list of vocab tuple(word,count)
    :param vector: a list of strings which are vectors
    :return:
    """
    with codecs.open(path,'w','utf8') as fp:
        fp.write(str(len(vocab)) + ' ' + str(dimension) + '\n')
        for i in range(len(vocab)):
            fp.write(vocab[i][0] + ' ' + ' '.join(vector[i]) + '\n')
    print("vector writing to {}".format(path))

FIX_LIMIT = 1000


vocab_fp = codecs.open(args.vocab_path, 'r', 'utf8')
vocab = vocab_fp.readlines()

model = gensim.models.Word2Vec.load("../../text.model")
dimension = model.vector_size




trainable_vocab_ls = []
trainable_vector_ls = []

fixed_vocab_ls = []
fixed_vector_ls = []

see_words_count = 0
count = 0

for word in vocab:

    word = word.strip('\n')

    try:
        result = model.wv[word]
        if see_words_count < FIX_LIMIT:
            see_words_count+=1

            trainable_vocab_ls.append((word,0))
            trainable_vector_ls.append(vector2str(result))
        else:
            fixed_vocab_ls.append( (word,0))
            fixed_vector_ls.append(vector2str(result))

    except:
        count += 1
        result = np.random.uniform(-0.1, 0.1, (dimension,))
        trainable_vocab_ls.append((word,0))
        trainable_vector_ls.append(vector2str(result))

print("fixed vocab:{},trainable vocab:{}".format(len(fixed_vocab_ls) , len(trainable_vocab_ls)))


save_vector(args.aim_path +'-trainable',trainable_vocab_ls,trainable_vector_ls,dimension)

save_vector(args.aim_path +'-fixed',fixed_vocab_ls, fixed_vector_ls,dimension)

trainable_vocab_ls.extend(fixed_vocab_ls) # now trainable_vocab_ls is the final vocab

save_vocab(args.vocab_path,trainable_vocab_ls,add_flag=False)


print("{}/{} words can't find the coresspending vectors.".format(count, len(trainable_vocab_ls)))
