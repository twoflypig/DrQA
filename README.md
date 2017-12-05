#  Introduce

This is a tensorflow version using DrQA model from  Reading Wikipedia to answer Opendomain questions

# Dependence:

​	jieba

​	tensorflow :1.3

​	hanziconv

# Note:

1. Where are NUM_TAG and EN_TAG used ?
  In make_vocab.py and  reader.py . Firstly,Using jieba to cut down the sentece ,where numbers and english words will be seperated, then use a regex to match en and numbers  in return list and also replaced them with tags. So when you want to change tags ,you should firstly look at this two files. 
2. test.py is used to confirm what is right or wrong


# Improvement:
1. Try use beam search rather than taking the arvmax
2. Use character embedding
3. ​


# Results 

​	on sougou qa set , word-level are 36% ac after 10 epochs

formate should like this 

```
.:
input  lib-word  modelRestor  one  output  __pycache__  readme.md  start.sh


./input:
data  data_2_3  train_factoid_2.json  train_factoid_3.json  valid  valid_factoid.json

./lib-word:
add_answer.py  make_vector.sh  nohup.out     pos_vocab          reader.py  test.ipynb   totokenize.py  ultize.py
evaluate.sh    make_vocab.py   parameter.py  produce_vector.py  README.md  test.py      train.py       ultize.pyc
inference.py   model           pos.py        __pycache__        tag_vocab  toSimply.py  train.sh       untitled.txt

./lib-word/model:
__init__.py  model_add_aligned.py  model_multiRNN.py  model.py  __pycache__

./output:
infer_answer.json  result  train_add_answer  train_Simply  valid_pred  vocab


```

