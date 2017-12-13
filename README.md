#  Introduce

This is a tensorflow version using DrQA model from  Reading Wikipedia to answer Opendomain questions

# Dependence:

​	jieba

​	tensorflow :1.3

# Data Form

A line is a json object, the form of json is like followings:

answer

passages:[]

​	'passage_id'

​	'passage_text'

​	'answer_p'

query:

query_id:

# Usage

## use pre-trained word-vector
If you want to be use pre-trained word-vector,make sure you have a model trained by gensim named text.model in ../../text.model .
then run 
```
bash make_vector.sh
```
This command will firstly make vocab from the training data,and then search text.model if the word can be found there.and the produce a vector named word-vector.
Next for training data, you should run
```
bash train.sh
```
After training ,you should run 
Next for training data, you should run
```
bash evaluate.sh
```
## the meaning of parameters
Here, we will explain the function of specific parameter.
    -- use_pretrain_vector . whether to use pre-trained word-vector,you should set use_pretrain_vector to be true and run make_vector.sh before training 
    -- pretrain_vector_split . whether to use fine-tuned word-vector,you should set use_pretrain_vector to be true and run make_vector.sh before training


# Note:

1. Where are NUM_TAG and EN_TAG used ?
  In make_vocab.py and  reader.py . Firstly,Using jieba to cut down the sentece ,where numbers and english words will be seperated, then use a regex to match en and numbers  in return list and also replaced them with tags. So when you want to change tags ,you should firstly look at this two files. 
2. test.py is used to confirm what is right or wrong


# Improvement:
1. Try use beam search rather than taking the arvmax
2. Use character embedding
3. ​I need to delete pos tag 


# Results 

on sougou qa set , word-level are 36% ac after 10 epochs

In Dec 6, model can achieve 47% ac after 20 epochs

In Dec 8, model can achieve 54% ac .

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

