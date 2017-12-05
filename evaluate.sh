#!/bin/bash
# firstly use regex to sub some unrelated words
# secondly,tokenize the passages
# thirdly, sent to model to process

python toSimply.py  --data_path ../input/valid --aim_path ../output/valid_pred

# produce the valid vocab

python make_vocab.py --add_answer False --src_path ../output/valid_pred --vocab_path ../output/infer_vocab

#python totokenize.py --data_path ../output/valid_pred --aim_path ../output/valid_tokenize

# in inference.py --is_training  default is false
python inference.py --vocab_path ../output/vocab --infer_vocab_path ../output/infer_vocab  --test_inference False  --data_path ../output/valid_pred \
	--use_pretrain_vector  false --vector_path word-vector --num_units 128 --input_embedding_size 200\
	   --keep_pro 1