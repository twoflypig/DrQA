#!/bin/bash
# this sh is for training 
# make vocab

rm -rf ../modelRestor/word-level
mkdir ../modelRestor/word-level

rm -rf ../output
mkdir ../output

# delete some unrelated characters
python toSimply.py --data_path ../input/data --aim_path ../output/train_Simply

python make_vocab.py  --src_path ../output/train_Simply  --vocab_path ../output/vocab

# add answer
python add_answer.py --src_path ../output/train_Simply --aim_path ../output/train_add_answer

# train
python train.py --vocab_path ../output/vocab --test train --data_path ../output/train_add_answer --is_training True
