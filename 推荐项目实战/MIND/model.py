import tensorflow as tf


class DNN(tf.keras.layers.Layer):
    """The Multi Layer Percetron
      Input shape
        - nD tensor with shape: ``(batch_size, ..., input_dim)``. The most common situation would be a 2D input with shape ``(batch_size, input_dim)``.
      Output shape
        - nD tensor with shape: ``(batch_size, ..., hidden_size[-1])``. For instance, for a 2D input with shape ``(batch_size, input_dim)``, the output would have shape ``(batch_size, hidden_size[-1])``.
      Arguments
        - **hidden_units**:list of positive integer, the layer number and units in each layer.
        - **activation**: Activation function to use.
        - **l2_reg**: float between 0 and 1. L2 regularizer strength applied to the kernel weights matrix.
        - **dropout_rate**: float in [0,1). Fraction of the units to dropout.
        - **use_bn**: bool. Whether use BatchNormalization before activation or not.
        - **seed**: A Python integer to use as random seed.
    """

    def __init__(self, hidden_units, activation='relu', l2_reg=0, dropout_rate=0, use_bn=False, seed=1024, **kwargs):
        self.hidden_units = hidden_units
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.seed = seed
        self.l2_reg = l2_reg
        self.use_bn = use_bn
        super(DNN, self).__init__(**kwargs)

    def build(self, input_shape):
        # if len(self.hidden_units) == 0:
        #     raise ValueError("hidden_units is empty")
        input_size = input_shape[-1]
        hidden_units = [int(input_size)] + list(self.hidden_units)
        self.kernels = [self.add_weight(name='kernel' + str(i),
                                        shape=(
                                            hidden_units[i], hidden_units[i + 1]),
                                        initializer=tf.keras.initializers.glorot_normal(
                                            seed=self.seed),
                                        regularizer=tf.keras.regularizers.l2(self.l2_reg),
                                        trainable=True) for i in range(len(self.hidden_units))]
        self.bias = [self.add_weight(name='bias' + str(i),
                                     shape=(self.hidden_units[i],),
                                     initializer=tf.keras.initializers.Zeros(),
                                     trainable=True) for i in range(len(self.hidden_units))]
        if self.use_bn:
            self.bn_layers = [tf.keras.layers.BatchNormalization() for _ in range(len(self.hidden_units))]

        self.dropout_layers = [tf.keras.layers.Dropout(self.dropout_rate, seed=self.seed + i) for i in
                               range(len(self.hidden_units))]

        self.activation_layers = [tf.keras.layers.Activation(self.activation) for _ in range(len(self.hidden_units))]

        super(DNN, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, inputs, training=None, **kwargs):

        deep_input = inputs

        for i in range(len(self.hidden_units)):
            fc = tf.nn.bias_add(tf.tensordot(
                deep_input, self.kernels[i], axes=(-1, 0)), self.bias[i])

            if self.use_bn:
                fc = self.bn_layers[i](fc, training=training)

            fc = self.activation_layers[i](fc)

            fc = self.dropout_layers[i](fc, training=training)
            deep_input = fc

        return deep_input

    def compute_output_shape(self, input_shape):
        if len(self.hidden_units) > 0:
            shape = input_shape[:-1] + (self.hidden_units[-1],)
        else:
            shape = input_shape

        return tuple(shape)

    def get_config(self, ):
        config = {'activation': self.activation, 'hidden_units': self.hidden_units,
                  'l2_reg': self.l2_reg, 'use_bn': self.use_bn, 'dropout_rate': self.dropout_rate, 'seed': self.seed}
        base_config = super(DNN, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class PoolingLayer(tf.keras.layers.Layer):

    def __init__(self, mode='mean', supports_masking=False, **kwargs):

        if mode not in ['sum', 'mean', 'max']:
            raise ValueError("mode must be sum or mean")
        self.mode = mode
        self.eps = tf.constant(1e-8, tf.float32)
        super(PoolingLayer, self).__init__(**kwargs)

        self.supports_masking = supports_masking

    def build(self, input_shape):

        super(PoolingLayer, self).build(
            input_shape)  # Be sure to call this somewhere!

    def call(self, seq_value_len_list, mask=None, **kwargs):
        if not isinstance(seq_value_len_list, list):
            seq_value_len_list = [seq_value_len_list]
        if len(seq_value_len_list) == 1:
            return seq_value_len_list[0]
        expand_seq_value_len_list = list(map(lambda x: tf.expand_dims(x, axis=-1), seq_value_len_list))
        a = tf.keras.layers.Concatenate(axis=-1)(expand_seq_value_len_list)
        if self.mode == "mean":
            hist = tf.math.reduce_mean(a, axis=-1, )
        if self.mode == "sum":
            hist = tf.math.reduce_sum(a, axis=-1, )
        if self.mode == "max":
            hist = tf.math.reduce_max(a, axis=-1, )
        return hist

    def get_config(self, ):
        config = {'mode': self.mode, 'supports_masking': self.supports_masking}
        base_config = super(PoolingLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def squash(inputs):
    vec_squared_norm = tf.math.reduce_sum(tf.square(inputs), axis=-1, keepdims=True)
    scalar_factor = vec_squared_norm / (1 + vec_squared_norm) / tf.sqrt(vec_squared_norm + 1e-8)
    vec_squashed = scalar_factor * inputs
    return vec_squashed


class CapsuleLayer(tf.keras.layers.Layer):
    def __init__(self, input_units, out_units, max_len, k_max, iteration_times=3,
                 init_std=1.0, **kwargs):
        self.input_units = input_units
        self.out_units = out_units
        self.max_len = max_len
        self.k_max = k_max
        self.iteration_times = iteration_times
        self.init_std = init_std
        super(CapsuleLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.routing_logits = self.add_weight(shape=[1, self.k_max, self.max_len],
                                              initializer=tf.keras.initializers.RandomNormal(stddev=self.init_std),
                                              trainable=False, name="B", dtype=tf.float32)
        self.bilinear_mapping_matrix = self.add_weight(shape=[self.input_units, self.out_units],
                                                       initializer=tf.keras.initializers.RandomNormal(
                                                           stddev=self.init_std),
                                                       name="S", dtype=tf.float32)
        super(CapsuleLayer, self).build(input_shape)

    def call(self, inputs, **kwargs):
        behavior_embddings, seq_len = inputs
        batch_size = tf.shape(behavior_embddings)[0]
        seq_len_tile = tf.tile(seq_len, [1, self.k_max])

        for i in range(self.iteration_times):
            mask = tf.sequence_mask(seq_len_tile, self.max_len)
            pad = tf.ones_like(mask, dtype=tf.float32) * (-2 ** 32 + 1)
            routing_logits_with_padding = tf.where(mask, tf.tile(self.routing_logits, [batch_size, 1, 1]), pad)
            weight = tf.nn.softmax(routing_logits_with_padding)
            behavior_embdding_mapping = tf.tensordot(behavior_embddings, self.bilinear_mapping_matrix, axes=1)
            Z = tf.matmul(weight, behavior_embdding_mapping) #error
            interest_capsules = squash(Z)
            delta_routing_logits = tf.math.reduce_sum(
                tf.matmul(interest_capsules, tf.transpose(behavior_embdding_mapping, perm=[0, 2, 1])),
                axis=0, keepdims=True
            )
            self.routing_logits.assign_add(delta_routing_logits)
        interest_capsules = tf.reshape(interest_capsules, [-1, self.k_max, self.out_units])
        return interest_capsules

    def compute_output_shape(self, input_shape):
        return (None, self.k_max, self.out_units)

    def get_config(self, ):
        config = {'input_units': self.input_units, 'out_units': self.out_units, 'max_len': self.max_len,
                  'k_max': self.k_max, 'iteration_times': self.iteration_times, "init_std": self.init_std}
        base_config = super(CapsuleLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class LabelAwareAttention(tf.keras.layers.Layer):
    def __init__(self, k_max, pow_p=1, **kwargs):
        self.k_max = k_max
        self.pow_p = pow_p
        super(LabelAwareAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        # Be sure to call this somewhere!

        self.embedding_size = input_shape[0][-1]
        super(LabelAwareAttention, self).build(input_shape)

    def call(self, inputs, training=None, **kwargs):
        keys = inputs[0]
        query = inputs[1]
        weight = tf.math.reduce_sum(keys * query, axis=-1, keepdims=True)
        weight = tf.math.pow(weight, self.pow_p)  # [x,k_max,1]

        if len(inputs) == 3:
            k_user = tf.cast(tf.maximum(
                1.,
                tf.minimum(
                    tf.cast(self.k_max, dtype="float32"),  # k_max
                    tf.log1p(tf.cast(inputs[2], dtype="float32")) / tf.log(2.)  # hist_len
                )
            ), dtype="int64")
            seq_mask = tf.transpose(tf.sequence_mask(k_user, self.k_max), [0, 2, 1])
            padding = tf.ones_like(seq_mask, dtype=tf.float32) * (-2 ** 32 + 1)  # [x,k_max,1]
            weight = tf.where(seq_mask, weight, padding)

        weight = tf.nn.softmax(weight, axis=1, name="weight")
        output = tf.math.reduce_sum(keys * weight, axis=1)

        return output

    def compute_output_shape(self, input_shape):
        return (None, self.embedding_size)

    def get_config(self, ):
        config = {'k_max': self.k_max, 'pow_p': self.pow_p}
        base_config = super(LabelAwareAttention, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class SampledSoftmaxLayer(tf.keras.layers.Layer):
    def __init__(self, num_sampled=5, **kwargs):
        self.num_sampled = num_sampled
        super(SampledSoftmaxLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.size = input_shape[0][0]
        self.zero_bias = self.add_weight(shape=[self.size],
                                         initializer=tf.keras.initializers.Zeros(),
                                         dtype=tf.float32,
                                         trainable=False,
                                         name="bias")
        super(SampledSoftmaxLayer, self).build(input_shape)

    def call(self, inputs_with_label_idx, training=None, **kwargs):
        """
        The first input should be the model as it were, and the second the
        target (i.e., a repeat of the training data) to compute the labels
        argument
        """
        embeddings, inputs, label_idx = inputs_with_label_idx
        loss = tf.nn.sampled_softmax_loss(weights=embeddings,  # self.item_embedding.
                                          biases=self.zero_bias,
                                          labels=label_idx,
                                          inputs=inputs,
                                          num_sampled=self.num_sampled,
                                          num_classes=self.size,  # self.target_song_size
                                          )
        return tf.expand_dims(loss, axis=1)

    def compute_output_shape(self, input_shape):
        return (None, 1)

    def get_config(self, ):
        config = {'num_sampled': self.num_sampled}
        base_config = super(SampledSoftmaxLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class FeatTile(tf.keras.layers.Layer):

    def __init__(self, k_max, **kwargs):
        self.k_max = k_max
        super(FeatTile, self).__init__(**kwargs)

    def build(self, input_shape):
        super(FeatTile, self).build(input_shape)

    def call(self, user_other_feature, **kwargs):
        return tf.tile(user_other_feature, [1, self.k_max, 1])  # 修改

    def get_config(self, ):
        config = {'k_max': self.k_max, }
        base_config = super(FeatTile, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class MIND(tf.keras.Model):
    def __init__(self, k_max, dynamic_k, embedding_dim=16, seq_max_len=20, p=1, num_sampled=10, feature_max_idx={},
                 **kwargs):
        super(MIND, self).__init__(**kwargs)
        self.k_max = k_max
        self.dynamic_k = dynamic_k
        self.embedding_dim = embedding_dim
        self.item_embedding = tf.Variable(
            tf.random.normal((feature_max_idx['movie_id'], self.embedding_dim), mean=0.0, stddev=0.05),
            name="item_embedding")
        self.user_embedding = tf.keras.layers.Embedding(input_dim=feature_max_idx['user_id'], output_dim=embedding_dim,
                                                        name="user_id")
        self.gender_embedding = tf.keras.layers.Embedding(input_dim=feature_max_idx['gender'], output_dim=embedding_dim,
                                                          name="gender")
        self.age_embedding = tf.keras.layers.Embedding(input_dim=feature_max_idx['age'], output_dim=embedding_dim,
                                                       name="age")
        self.occupation_embedding = tf.keras.layers.Embedding(input_dim=feature_max_idx['occupation'],
                                                              output_dim=embedding_dim,
                                                              name="occupation")
        self.zip_embedding = tf.keras.layers.Embedding(input_dim=feature_max_idx['zip'], output_dim=embedding_dim,
                                                       name="zip")

        self.PoolingLayer = PoolingLayer()
        self.CapsuleLayer = CapsuleLayer(input_units=embedding_dim, out_units=embedding_dim, max_len=seq_max_len,
                                         k_max=self.k_max)
        self.FeatTile = FeatTile(self.k_max)
        self.SampledSoftmaxLayer = SampledSoftmaxLayer(num_sampled=num_sampled)
        self.LabelAwareAttention = LabelAwareAttention(k_max=k_max, pow_p=p)
        self.DNN = DNN((64, self.embedding_dim), 'relu', 0, 0, False, 1024, name="user_embedding")

    def call(self, inputs, training=None):

        user_id, gender, age, occupation, zip, movie_id, hist_movie_id, hist_len = inputs['user_id'], inputs['gender'], \
                                                                                   inputs[
                                                                                       'age'], inputs['occupation'], \
                                                                                   inputs['zip'], inputs['movie_id'], inputs[
                                                                                       'hist_movie_id'], inputs[
                                                                                       'hist_len']
        ## 如何user_id 维度为(None,) 则需要扩展维度(None,1)
        user_id = tf.expand_dims(user_id,axis=1)
        gender = tf.expand_dims(gender, axis=1)
        age = tf.expand_dims(age, axis=1)
        occupation = tf.expand_dims(occupation, axis=1)
        zip = tf.expand_dims(zip, axis=1)
        movie_id = tf.expand_dims(movie_id, axis=1)
        hist_len = tf.expand_dims(hist_len, axis=1)

        user_emb = self.user_embedding(user_id)
        gender_emb = self.gender_embedding(gender)
        age_emb = self.age_embedding(age)
        occupation_emb = self.occupation_embedding(occupation)
        zip_emb = self.zip_embedding(zip)
        item_emb = tf.nn.embedding_lookup(self.item_embedding, movie_id)
        hist_item_emb = tf.nn.embedding_lookup(self.item_embedding, hist_movie_id)

        #
        # item = tf.expand_dims(movie_id, axis=1)
        # hist_len = tf.expand_dims(hist_len, axis=1)

        user_other_features = self.PoolingLayer(
            [user_emb, gender_emb, gender_emb, age_emb, occupation_emb, zip_emb])

        target_emb = self.PoolingLayer([item_emb])
        history_emb = self.PoolingLayer([hist_item_emb])
        other_feature_tile = self.FeatTile(user_other_features)


        high_capsule = self.CapsuleLayer((history_emb, hist_len))

        user_deep_input = tf.keras.layers.Concatenate(axis=-1)([other_feature_tile, high_capsule])

        user_embeddings = self.DNN(user_deep_input)  # return user embedding

        if self.dynamic_k:
            user_embedding_final = self.LabelAwareAttention((user_embeddings, target_emb, hist_len))
        else:
            user_embedding_final = self.LabelAwareAttention((user_embeddings, target_emb))

        output = self.SampledSoftmaxLayer([self.item_embedding, user_embedding_final, movie_id])  # loss
        return output
        # if training == True:
        #     return output
        # else:
        #     return user_embeddings, self.item_embedding

#             logits = tf.matmul(user_embedding_final, tf.transpose(self.item_embedding))
#             labels_one_hot = tf.one_hot(item, 2)
#             loss = tf.nn.softmax_cross_entropy_with_logits(labels=labels_one_hot, logits=logits)

#         return output
#         if training==True:
#             return output
#         else:
#             pred  = tf.reduce_max(tf.squeeze(tf.matmul(user_embeddings,item_emb,transpose_b=True),axis=-1),axis=1)
#             return pred
