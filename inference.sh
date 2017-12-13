#!/bin/bash
python ./others/professiion.py
python ./ultize/toSimply.py  --data_path ../output/self-defined.json --aim_path ../output/self_defined_pred
python inference.py --vocab_path ../output/vocab --test inference --data_path ../output/self_defined_pred 	--use_pretrain_vector  false --vector_path word-vector --num_units 128 --input_embedding_size 200 --keep_pro 1 --show_self_define True