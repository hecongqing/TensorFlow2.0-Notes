"""
https://github.com/alibaba/esim-response-selection
"""
import tensorflow as tf


class bilstm(tf.keras.layers.Layer):
    def __init__(self, units=50, **kwargs):
        self.lstm_fw = tf.keras.layers.RNN(tf.keras.layers.LSTMCell(units), return_sequences=True, go_backwards=False)
        self.lstm_bw = tf.keras.layers.RNN(tf.keras.layers.LSTMCell(units), return_sequences=True, go_backwards=True)
        super(bilstm, self).__init__(**kwargs)

    def call(self, x):
        x_lstm_fw = self.lstm_fw(x)
        x_lstm_bw = self.lstm_bw(x)
        x_cmp = tf.keras.layers.concatenate([x_lstm_fw, x_lstm_bw])
        return x_cmp
    def get_config(self, ):
        config = {'units': self.units}
        base_config = super(bilstm, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class esim(tf.keras.Model):
    def __init__(self, rate=0.1, num_labels=2, **kwargs):
        super(esim, self).__init__(**kwargs)
        self.dropout = tf.keras.layers.Dropout(rate=rate)
        self.embed = tf.keras.layers.Embedding(input_dim=20000, output_dim=50, embeddings_initializer='uniform')
        self.bilstm1 = bilstm(units=50)
        self.bilstm2 = bilstm(units=50)
        self.local_inference_ = local_inference()
        self.dense = tf.keras.layers.Dense(100, activation=tf.nn.relu,
                                           kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02))
        self.dense1 = tf.keras.layers.Dense(100, activation=tf.nn.tanh,
                                            kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02))
        self.outputs = tf.keras.layers.Dense(num_labels, activation=None,
                                             kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02))

    def call(self, inputs, training=None):
        x1, x2 = inputs

        x1_mask = 1 - tf.dtypes.cast(tf.equal(x1, 0), tf.float32)
        x2_mask = 1 - tf.dtypes.cast(tf.equal(x2, 0), tf.float32)
        emb1 = self.embed(x1)
        emb2 = self.embed(x2)

        emb1 = self.dropout(emb1, training=training)
        emb2 = self.dropout(emb2, training=training)

        emb1 = emb1 * tf.expand_dims(x1_mask, -1)
        emb2 = emb2 * tf.expand_dims(x2_mask, -1)

        # 第一次bilstm
        x1_enc = self.bilstm1(emb1)
        x2_enc = self.bilstm1(emb2)

        # attention交互
        x1_enc = x1_enc * tf.expand_dims(x1_mask, -1)
        x2_enc = x2_enc * tf.expand_dims(x2_mask, -1)
        x1_dual, x2_dual = self.local_inference_(x1_enc, x1_mask, x2_enc, x2_mask)

        x1_match = tf.keras.layers.concatenate([x1_enc, x1_dual, x1_enc * x1_dual, x1_enc - x1_dual], axis=2)
        x2_match = tf.keras.layers.concatenate([x2_enc, x2_dual, x2_enc * x2_dual, x2_enc - x2_dual], axis=2)

        x1_match_mapping = self.dense(x1_match)
        x2_match_mapping = self.dense(x2_match)
        x1_match_mapping = self.dropout(x1_match_mapping, training=training)
        x2_match_mapping = self.dropout(x2_match_mapping, training=training)

        # 第二次bilstm
        x1_cmp = self.bilstm2(x1_match_mapping)
        x2_cmp = self.bilstm2(x2_match_mapping)

        # pooling 操作
        logit_x1_sum = tf.math.reduce_sum(x1_cmp * tf.expand_dims(x1_mask, -1), axis=1) / tf.expand_dims(
            tf.reduce_sum(x1_mask, axis=1), axis=1)
        logit_x1_max = tf.math.reduce_max(x1_cmp * tf.expand_dims(x1_mask, -1), axis=1)
        logit_x2_sum = tf.math.reduce_sum(x2_cmp * tf.expand_dims(x2_mask, -1), axis=1) / tf.expand_dims(
            tf.reduce_sum(x2_mask, axis=1), axis=1)
        logit_x2_max = tf.math.reduce_max(x2_cmp * tf.expand_dims(x2_mask, -1), axis=1)

        logit = tf.keras.layers.concatenate([logit_x1_sum, logit_x1_max, logit_x2_sum, logit_x2_max], axis=1)
        # final classifier
        logit = self.dropout(logit, training=training)
        logit = self.dense1(logit)
        logit = self.outputs(logit)
        probability = tf.nn.softmax(logit)
        return probability


class local_inference(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(local_inference, self).__init__(**kwargs)

    def call(self, x1, x1_mask, x2, x2_mask):
        """Local inference collected over sequences
        Args:
            x1: float32 Tensor of shape [seq_length1, batch_size, dim].
            x1_mask: float32 Tensor of shape [seq_length1, batch_size].
            x2: float32 Tensor of shape [seq_length2, batch_size, dim].
            x2_mask: float32 Tensor of shape [seq_length2, batch_size].
        Return:
            x1_dual: float32 Tensor of shape [seq_length1, batch_size, dim]
            x2_dual: float32 Tensor of shape [seq_length2, batch_size, dim]
        """

        # x1: [batch_size, seq_length1, dim].
        # x1_mask: [batch_size, seq_length1].
        # x2: [batch_size, seq_length2, dim].
        # x2_mask: [batch_size, seq_length2].
        x1 = tf.transpose(x1, [1, 0, 2])
        x1_mask = tf.transpose(x1_mask, [1, 0])
        x2 = tf.transpose(x2, [1, 0, 2])
        x2_mask = tf.transpose(x2_mask, [1, 0])
        # attention_weight: [batch_size, seq_length1, seq_length2]
        attention_weight = tf.matmul(x1, tf.transpose(x2, [0, 2, 1]))
        # calculate normalized attention weight x1 and x2
        # attention_weight_2: [batch_size, seq_length1, seq_length2]
        attention_weight_2 = tf.math.exp(
            attention_weight - tf.math.reduce_max(attention_weight, axis=2, keepdims=True))
        attention_weight_2 = attention_weight_2 * tf.expand_dims(x2_mask, 1)
        # alpha: [batch_size, seq_length1, seq_length2]
        alpha = attention_weight_2 / (tf.math.reduce_sum(attention_weight_2, -1, keepdims=True) + 1e-8)
        # x1_dual: [batch_size, seq_length1, dim]
        x1_dual = tf.math.reduce_sum(tf.expand_dims(x2, 1) * tf.expand_dims(alpha, -1), 2)
        # x1_dual: [seq_length1, batch_size, dim]
        x1_dual = tf.transpose(x1_dual, [1, 0, 2])

        # attention_weight_1: [batch_size, seq_length2, seq_length1]
        attention_weight_1 = attention_weight - tf.math.reduce_max(attention_weight, axis=1, keepdims=True)
        attention_weight_1 = tf.math.exp(tf.transpose(attention_weight_1, [0, 2, 1]))
        attention_weight_1 = attention_weight_1 * tf.expand_dims(x1_mask, 1)

        # beta: [batch_size, seq_length2, seq_length1]
        beta = attention_weight_1 / (tf.math.reduce_sum(attention_weight_1, -1, keepdims=True) + 1e-8)
        # x2_dual: [batch_size, seq_length2, dim]
        x2_dual = tf.math.reduce_sum(tf.expand_dims(x1, 1) * tf.expand_dims(beta, -1), 2)
        # x2_dual: [seq_length2, batch_size, dim]
        x2_dual = tf.transpose(x2_dual, [1, 0, 2])
        return x1_dual, x2_dual
    def get_config(self, ):
        config = {}
        base_config = super(local_inference, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
