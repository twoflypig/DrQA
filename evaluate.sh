#!/bin/bash
# firstly use regex to sub some unrelated words
# secondly,tokenize the passages
# thirdly, sent to model to process


# Step1:delete some unrelated characters
python -m ultize.toSimply  --data_path ../input/valid --aim_path ../output/valid_pred
# Step2: produce vocab
python -m ultize.make_vocab --add_answer False --src_path ../output/valid_pred --vocab_path ../output/infer_vocab
# Step3: tokenize files
python -m ultize.totokenize --data_path ../output/valid_pred --aim_path ../output/valid_tokenize
# Step4: inference process
python inference.py --vocab_path ../output/vocab --infer_vocab_path ../output/infer_vocab  --test_inference False  --data_path ../output/valid_tokenize \
	--use_pretrain_vector  true --pretrain_vector_split true  --vector_path word-vector  --num_units 128 --input_embedding_size 200\
	   --keep_pro 1