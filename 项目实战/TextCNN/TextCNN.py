#!/usr/bin/env python 
# -*- coding:utf-8 -*-
"""
Author:
    Congqing He,hecongqing@hotmail.com
"""

import tensorflow as tf

print(tf.__version__)


# class PredictionLayer(tf.keras.layers.Layer):
#     def __init__(self):

###input 部分写的有问题
class InputLayer(tf.keras.layers.Layer):
    def __init__(self, vocab_num=20000, sequence_length=10, name=None, **kwargs):
        self.vocab_num = vocab_num
        self.sequence_length = sequence_length
        super(InputLayer, self).__init__(name=name, **kwargs)

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        input_ = tf.strings.split(inputs, sep=' ', name='input_split')
        input_ = tf.strings.to_hash_bucket(input_, num_buckets=self.vocab_num, name='input_hash')
        input_ = tf.squeeze(input_, axis=1)
        input_ = input_.to_tensor(default_value=0, name='input_totensor')
        input_ = tf.RaggedTensor.from_tensor(input_, lengths=[self.sequence_length] * batch_size,
                                             name='input_from_tensor')
        input_ = input_.to_tensor(default_value=0, name='input_totensor2')
        return input_

    def get_config(self):
        config = {'vocab_num': self.vocab_num, 'sequence_length': self.sequence_length}
        base_config = super(InputLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class CNNEncoder(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, strides, padding, activation, name='cnnencoder', **kwargs):
        super(CNNEncoder, self).__init__(name=name, **kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.activation = activation

    def build(self, input_shape):
        self.cnn_layers = []
        self.max_poolings = []
        for i, _ in enumerate(self.kernel_size):
            self.cnn_layers.append(
                tf.keras.layers.Convolution1D(filters=self.filters, kernel_size=_, strides=self.strides,
                                              padding=self.padding, activation=self.activation,
                                              name='cnn_{0}'.format(str(i)))
            )
            self.max_poolings.append(tf.keras.layers.GlobalAveragePooling1D(name='pooling_{0}'.format(str(i))))
        super(CNNEncoder, self).build(input_shape)

    def call(self, inputs):
        feat_concat = []
        for i in range(len(self.kernel_size)):
            cnnlayer = self.cnn_layers[i](inputs)
            cnnlayer = self.max_poolings[i](cnnlayer)
            feat_concat.append(cnnlayer)
        return tf.keras.layers.concatenate(feat_concat, axis=-1)

    def get_config(self):
        config = {'filters': self.filters,
                  'kernel_size': self.kernel_size,
                  'strides': self.strides,
                  'padding': self.padding,
                  'activation': self.activation
                  }
        base_config = super(CNNEncoder, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class DenseLayer(tf.keras.layers.Layer):
    def __init__(self, hidden_units=[128, 64], use_bn=True, dropout_rate=0.1, name=None, **kwargs):
        self.hidden_units = hidden_units
        self.use_bn = use_bn
        self.dropout_rate = dropout_rate
        self.activate = tf.keras.layers.Activation(activation='relu')
        super(DenseLayer, self).__init__(name=name, **kwargs)

    def build(self, input_shape):
        self.denselayers = []
        # self.batchnorm =tf.keras.layers.BatchNormalization(axis=-1)
        for hidden_unit in self.hidden_units:
            self.denselayers.append(tf.keras.layers.Dense(hidden_unit, name='dense_{0}'.format(str(hidden_unit))))
            # self.batchnormlayers.append(tf.keras.layers.BatchNormalization(axis=-1,name='bn_{0}'.format(str(hidden_unit))))
        super(DenseLayer, self).build(input_shape)

    def call(self, inputs):
        fc = inputs
        for i, denselayer in enumerate(self.denselayers):
            fc = denselayer(fc)
            fc = self.activate(fc)
            fc = tf.nn.dropout(fc, self.dropout_rate)
            # fc = self.batchnormlayers[i](fc)
            # fc = self.batchnorm(fc)
        return fc

    def get_config(self):
        config = {'hidden_units': self.hidden_units, 'use_bn': self.use_bn,
                  'dropout_rate': self.dropout_rate}
        base_config = super(DenseLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class TextCNN(tf.keras.Model):
    def __init__(self, label_num, **kwargs):
        self.label_num = label_num
        super(TextCNN, self).__init__(name='TextCNN', **kwargs)

    def build(self, input_shape):
        self.Embedding = tf.keras.layers.Embedding(input_dim=20,
                                                   output_dim=10,
                                                   embeddings_initializer='uniform',
                                                   mask_zero=False,
                                                   trainable=True,
                                                   name="embedding")
        self.InputLayer = InputLayer(vocab_num=20, sequence_length=10, name='InputLayer')
        self.CNNEncoder = CNNEncoder(filters=64, kernel_size=[2, 3, 4], strides=1, padding='same',
                                     activation='relu', name='CNNEncoder')
        self.DenseLayer = DenseLayer(hidden_units=[256, 128, 64], use_bn=True, dropout_rate=0.1, name='DenseLayer')
        self.OutputLayer = tf.keras.layers.Dense(self.label_num, name='output')
        super(TextCNN, self).build(input_shape)

    # @tf.function(input_signature=[tf.TensorSpec([None], tf.string)])
    @tf.function
    def call(self, inputs):
        x = self.InputLayer(inputs)
        x = self.Embedding(x)
        x = self.CNNEncoder(x)
        fc = self.DenseLayer(x)
        fc = self.OutputLayer(fc)
        fc = tf.nn.softmax(fc, name='Probability')
        return fc


import numpy as np
import pandas as pd

# train_df = pd.read_csv("../data/cnews/train.tsv", header=None, names=['label', 'content'])
# val_df = pd.read_csv("../data/cnews/train.tsv", header=None, names=['label', 'content'])
# test_df = pd.read_csv("../data/cnews/train.tsv", header=None, names=['label', 'content'])

data_x = np.array(["你好 我好 大家 好", "你好 我好 大家 好", "你好 我好 大家 好 好 好 好 好 好 好 好 好 好 好 好 好"]).reshape((-1, 1))
data_y = np.array([0, 1, 2]).reshape((-1, 1))
model = TextCNN(label_num=3)

model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss=tf.keras.losses.CategoricalCrossentropy(),
              metrics=['accuracy'])
data_y = tf.keras.utils.to_categorical(data_y)
model.fit(data_x, data_y, epochs=1, batch_size=1, )
model.summary()

tf.saved_model.save(model, './save_models/textcnn1/')
infer_model = tf.saved_model.load('./save_models/textcnn1/')
infer = infer_model.signatures["serving_default"]
print(infer(tf.constant([['你好 我好 大家 好'], ['你好 我好 大家 好']])))

# tf.keras.models.save_model(model, './save_models/textcnn1/', save_format="tf")
# model_new = tf.keras.models.load_model('./save_models/textcnn1/')
# print(model_new.predict([['你好 我好 大家 好']]))
