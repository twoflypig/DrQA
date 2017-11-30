#!/bin/bash
# firstly use regex to sub some unrelated words
# secondly,tokenize the passages
# thirdly, sent to model to process

python toSimply.py  --data_path ../input/valid --aim_path ../output/valid_pred

# produce the valid vocab

#python make_vocab.py --add_answer False --src_path ../input/valid --vocab_path ../output/valid_vocab

#python totokenize.py --data_path ../output/valid_pred --aim_path ../output/valid_tokenize

# in inference.py --is_training  default is false
python inference.py --vocab_path ../output/vocab --test inference --data_path ../output/valid_pred \
	--use_pretrain_vector  false --vector_path word-vector --num_units 128 --input_embedding_size 200\
	--keep_pro 1