#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function



import base64
import json
from io import BytesIO
import tensorflow as tf
import jieba
import numpy as np
import requests
from flask import Flask, request, jsonify
import pickle


app = Flask(__name__)




# Testing URL
@app.route('/cls/', methods=['POST'])
def textclassification_cls():
    sentence = request.get_data()
    sentence = " ".join(jieba.lcut(sentence))
    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
    #编码
    sentence_seq = tokenizer.texts_to_sequences([sentence])
    #填充
    if len(sentence_seq[0])<800:
        X_new = [sentence_seq[0]+[0]*(800-len(sentence_seq[0]))]
    else:

        X_new = [sentence_seq[0][:800]]


    input_data_json = json.dumps({
        "signature_name": "serving_default",
        "instances": [{"contents":X_new[0]}],
    })
    SERVER_URL = 'http://192.168.0.104:8501/v1/models/textclassification:predict'
    response = requests.post(SERVER_URL, data=input_data_json)
    response = json.loads(response.text)
    predictions = response['predictions'][0]
    # 体育

    return str(predictions)

if __name__ == '__main__':
    app.run('0.0.0.0',port=5000,debug=True)