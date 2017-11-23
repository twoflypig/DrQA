import codecs
import json
from ultize import *
### code 1:
# with codecs.open("pos_vocab","r","utf8") as fp:
# 	with codecs.open("pos_vocab_new","w","utf8") as aim_fp:
# 		for line in fp:
# 			temp = line.strip().split()
# 			aim_fp.write(temp[0]
# 				+'\n')


### code 2:converting 

# pos_ls = []
# with codecs.open("../output/train_add_answer","r","utf8") as fp:
# 	for index,line in enumerate(fp):
# 		line = json.loads(line)

# 		query , query_pos  =   token_pos( line['query'])

# 		pos_ls.extend(query_pos)
		
# 		for i in range(len(line['passages'])):
# 			# passage
# 			passage , passage_pos   =  token_pos (line['passages'][i]['passage_text'] )

# 			pos_ls.extend(passage_pos)
# 		if index %100 ==0:
# 			print(index)

# pos_ls = set(pos_ls)

# with codecs.open("pos_vocab_new","w","utf8") as aim_fp:
# 	for item in pos_ls:
# 		aim_fp.write(item +'\n')

### code 3:
# 把两个vocab合并
def get_set(fp):
	data = []
	for line in fp:
		data.append(line.strip())
	data = set(data)
	return data
with codecs.open("pos_vocab","r","utf8") as fp:
	with codecs.open("pos_vocab_new","r","utf8") as fp2:
		with codecs.open("pos_vocab_set","w","utf8") as aim_fp:

			vocab1 = get_set(fp)
			vocab2 = get_set(fp2)
			vocab = set(vocab2 | vocab1)
			print("vocba1:{}".format(len(vocab1)))
			print("vocba2:{}".format(len(vocab2)))
			print("vocba:{}".format(len(vocab)))
			for item in vocab:
				aim_fp.write(item +'\n')