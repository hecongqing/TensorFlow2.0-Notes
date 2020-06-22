# -*- coding: utf-8 -*-
import json
import os
import datetime as dt
from tqdm import tqdm
import pandas as pd
import numpy as np

#label
def f2cat(filename: str) -> str:
    return filename.split('.')[0]

#读取文件、label变成list
class Simplified():
    def __init__(self, input_path='./input'):
        self.input_path = input_path

    def list_all_categories(self):
        files = os.listdir(self.input_path)
        return sorted([f2cat(f) for f in files], key=str.lower)

    def read_training_csv(self, category, nrows=None, usecols=None, drawing_transform=False):
        df = pd.read_csv(os.path.join(self.input_path, category + '.csv'),
                         nrows=nrows, parse_dates=['timestamp'], usecols=usecols)
        if drawing_transform:
            df['drawing'] = df['drawing'].apply(json.loads)
        return df

start = dt.datetime.now()
#加载数据路径
s = Simplified('./data/')
NCSVS = 100
categories = s.list_all_categories()
print(len(categories))

for y, cat in tqdm(enumerate(categories)):
    df = s.read_training_csv(cat)
    df['y'] = y
    df['cv'] = (df.key_id // 10 ** 7) % NCSVS #100 
    for k in range(NCSVS):
        filename = './shuffle_data/train_k{}.csv'.format(k)
        chunk = df[df.cv == k]
        chunk = chunk.drop(['key_id'], axis=1)
        if y == 0:
            chunk.to_csv(filename, index=False)
        else:
            chunk.to_csv(filename, mode='a', header=False, index=False)

#对csv数据打乱，进行压缩
for k in tqdm(range(NCSVS)):
    filename = './shuffle_data/train_k{}.csv'.format(k)
    if os.path.exists(filename):
        df = pd.read_csv(filename)
        df['rnd'] = np.random.rand(len(df))
        df = df.sort_values(by='rnd').drop('rnd', axis=1)
        df.to_csv('./shuffle_data_gzip/train_k{}.csv.gz'.format(k), compression='gzip', index=False)
        os.remove(filename)
print(df.shape)

end = dt.datetime.now()
print('Latest run {}.\nTotal time {}s'.format(end, (end - start).seconds))
