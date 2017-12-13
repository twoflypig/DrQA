#!/bin/bash
# this sh is for training 
# make vocab

rm -rf ../modelRestor/word-level
mkdir ../modelRestor/word-level

rm -rf ../output
mkdir ../output

# Step1:delete some unrelated characters
echo "Step1:delete some unrelated characters"
python ./ultize/toSimply.py --data_path ../input/data --aim_path ../output/train_Simply --process_answer True
# Step2: produce vocab
echo "Step2: produce vocab"
python ./ultize/make_vocab.py  --src_path ../output/train_Simply  --vocab_path ../output/vocab --add_answer True
# Step3: adding answer
echo "Step3: adding answer"
python ./ultize/add_answer.py --src_path ../output/train_Simply --aim_path ../output/train_add_answer
# Step4: tokenize files
echo "Step4: tokenize files"
python ./ultize/totokenize.py --data_path  ../output/train_add_answer  --aim_path  ../output/train_add_answer_split
# Step5: training process
echo "Step5: training process"
python train.py --vocab_path ../output/vocab --test train --data_path ../output/train_add_answer_split --is_training True\
     --use_pretrain_vector  false --pretrain_vector_split true --vector_path word-vector --num_units 128 --input_embedding_size 200 --batch_size 20\
     --keep_pro 0.6

# test
# python test.py --vocab_path ../output/vocab --test train --data_path ../output/train_add_answer --is_training True
#      --use_pretrain_vector  false --vector_path word-vector --num_units 128 --input_embedding_size 200 --batch_size 20
#      --keep_pro 0.7
