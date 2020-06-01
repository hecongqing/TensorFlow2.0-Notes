#!/usr/bin/env python 
# -*- coding:utf-8 -*-
"""
Author:
    Congqing He,hecongqing@hotmail.com
"""

import pandas as pd
import jieba
train_df = pd.read_csv("../data/cnews/train.tsv", sep='\t', header=None, names=['label', 'content'])
val_df = pd.read_csv("../data/cnews/train.tsv", sep='\t', header=None, names=['label', 'content'])
test_df = pd.read_csv("../data/cnews/train.tsv", sep='\t', header=None, names=['label', 'content'])

#分词

print(train_df.head())
