import gensim
import codecs
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='produce answer and passage vocab.')

parser.add_argument('--src_path',type= str ,default = "None",
                    help='the origin data path')
parser.add_argument('--vocab_path', type= str, default = "None",
                    help='the path of vocab')
parser.add_argument('--aim_path', type= str, default = "None",
                    help='whether to add answer vocab')
args = parser.parse_args()

aim_fp = codecs.open(args.aim_path,'w','utf8')
vocab_fp  = codecs.open(args.vocab_path,'r','utf8')
vocab =  vocab_fp.readlines()


model = gensim.models.Word2Vec.load("../../text.model")
dimension = model.vector_size

aim_fp.write( str(len(vocab)) +' ' +str(dimension) +'\n')

count = 0 
for word in vocab:

	word= word.strip('\n')
	#print(word)
	
	try:
		result = model.wv[word]
	except:
		count +=1
		result = np.random.uniform(-0.1,0.1,(dimension,))
	result = result.tolist()
	result = [str(x) for x in result]
	aim_fp.write(word+' '+' '.join(result) +'\n')

print("{}/{} words can't find the coresspending vectors.".format(count,len(vocab)))

