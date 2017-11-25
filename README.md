#  Introduce

This is a tensorflow version using DrQA model from  Reading Wikipedia to answer Opendomain questions

# Dependence:

​	jieba

​	tensorflow :1.3

​	hanziconv

​	

# Results 

​	on sougou qa set , word-level are 36% ac after 10 epochs

formate should like this 

```
├── input
│   ├── data
│   └── valid
├── lib-word
│   ├── add_answer.py
│   ├── evaluate.sh
│   ├── filter.py
│   ├── inference.py
│   ├── make_vocab.py
│   ├── model
│   │   ├── __init__.py
│   │   ├── model_add_aligned.py
│   │   ├── model_multiRNN.py
│   │   ├── model.py
│   │   └── __pycache__
│   │       ├── __init__.cpython-36.pyc
│   │       ├── model_add_aligned.cpython-36.pyc
│   │       └── model.cpython-36.pyc
│   ├── pos.py
│   ├── pos_vocab
│   ├── process.py
│   ├── __pycache__
│   │   ├── reader.cpython-36.pyc
│   │   ├── tokenize.cpython-36.pyc
│   │   └── ultize.cpython-36.pyc
│   ├── reader.py
│   ├── README.md
│   ├── sougou
│   ├── toSimply.py
│   ├── totokenize.py
│   ├── train.py
│   ├── train.sh
│   ├── ultize.py
│   └── untitled.txt
├── modelRestor
│   ├── model_base
│   │   ├── checkpoint
│   │   ├── model.ckpt-49716.data-00000-of-00001
│   │   ├── model.ckpt-49716.index
│   │   └── model.ckpt-49716.meta
│   ├── model_base_v2
│   │   ├── checkpoint
│   │   ├── events.out.tfevents.1508772930.lival
│   │   ├── model.ckpt-74503.data-00000-of-00001
│   │   ├── model.ckpt-74503.index
│   │   └── model.ckpt-74503.meta
│   └── word-level
│       ├── -41602.data-00000-of-00001
│       ├── -41602.index
│       ├── -41602.meta
│       ├── -43595.data-00000-of-00001
│       ├── -43595.index
│       ├── -43595.meta
│       ├── -44566.data-00000-of-00001
│       ├── -44566.index
│       ├── -44566.meta
│       ├── -46560.data-00000-of-00001
│       ├── -46560.index
│       ├── -46560.meta
│       ├── -48554.data-00000-of-00001
│       ├── -48554.index
│       ├── -48554.meta
│       ├── checkpoint
│       └── events.out.tfevents.1511419759.Zehao-Lab
├── output
│   ├── assembled_result
│   ├── base
│   │   ├── nn_result
│   │   └── result
│   ├── filter-data
│   ├── nn_result
│   ├── pre_processed
│   ├── result
│   ├── tfidf_result
│   ├── train_add_answer
│   ├── train_answer_vocab
│   ├── train_Simply
│   ├── valid_pred
│   ├── valid_tokenize
│   ├── valid_vocab
│   └── vocab

```

