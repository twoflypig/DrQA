#!/bin/bash
# only used in an remote machine named 199

# produce vocab
python ./ultize/make_vocab.py  --src_path ../output/train_Simply  --vocab_path ../output/vocab

# produce vector according to vocab
python ./ultize/produce_vector.py --src_path ../../text.model --vocab_path ../output/vocab --aim_path word-vector