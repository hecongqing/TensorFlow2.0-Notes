import os
import time

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_datasets as tfds

print(tf.__version__)

train = pd.read_csv("./nmt/news-commentary-v14.en-zh.tsv", error_bad_lines=False, sep='\t', header=None)

train_df = train.iloc[:280000]
val_df = train.iloc[280000:]

with tf.io.TFRecordWriter('./nmt/train.tfrecord') as writer:
    for en, zh in train_df.values:
        try:
            feature = {  # 建立 tf.train.Feature 字典
                'en': tf.train.Feature(bytes_list=tf.train.BytesList(value=[str.encode(en)])),
                'zh': tf.train.Feature(bytes_list=tf.train.BytesList(value=[str.encode(zh)]))
            }

            example = tf.train.Example(features=tf.train.Features(feature=feature))  # 通过字典建立 Example
            writer.write(example.SerializeToString())  # 将Example序列化并写入 TFRecord 文件
        except:
            pass

with tf.io.TFRecordWriter('./nmt/valid.tfrecord') as writer:
    for en, zh in val_df.values:
        try:
            feature = {  # 建立 tf.train.Feature 字典
                'en': tf.train.Feature(bytes_list=tf.train.BytesList(value=[str.encode(en)])),
                'zh': tf.train.Feature(bytes_list=tf.train.BytesList(value=[str.encode(zh)]))
            }

            example = tf.train.Example(features=tf.train.Features(feature=feature))  # 通过字典建立 Example
            writer.write(example.SerializeToString())  # 将Example序列化并写入 TFRecord 文件
        except:
            pass

feature_description = {  # 定义Feature结构，告诉解码器每个Feature的类型是什么
    'en': tf.io.FixedLenFeature([], tf.string),
    'zh': tf.io.FixedLenFeature([], tf.string),
}


def _parse_example(example_string):  # 将 TFRecord 文件中的每一个序列化的 tf.train.Example 解码
    feature_dict = tf.io.parse_single_example(example_string, feature_description)
    return feature_dict['en'], feature_dict['zh']


train_examples = tf.data.TFRecordDataset('./nmt/train.tfrecord').map(_parse_example)
val_examples = tf.data.TFRecordDataset('./nmt/valid.tfrecord').map(_parse_example)

output_dir = "./nmt"
en_vocab_file = os.path.join(output_dir, "en_vocab")
zh_vocab_file = os.path.join(output_dir, "zh_vocab")
checkpoint_path = os.path.join(output_dir, "checkpoints")
log_dir = os.path.join(output_dir, 'logs')

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

try:
    subword_encoder_en = tfds.features.text.SubwordTextEncoder.load_from_file(en_vocab_file)
    print(f"载入已建立的字典： {en_vocab_file}")
except:
    print("没有已建立的字典，从头建立。")
    subword_encoder_en = tfds.features.text.SubwordTextEncoder.build_from_corpus(
        (en.numpy() for en, _ in train_examples),
        target_vocab_size=10000)  # 有需要可以调整字典大小

    # 将字典档案存下以方便下次 warmstart
    subword_encoder_en.save_to_file(en_vocab_file)

print(f"字典大小：{subword_encoder_en.vocab_size}")
print(f"前 10 个 subwords：{subword_encoder_en.subwords[:10]}")
print()

try:
    subword_encoder_zh = tfds.features.text.SubwordTextEncoder.load_from_file(zh_vocab_file)
    print(f"载入已建立的字典： {zh_vocab_file}")
except:
    print("没有已建立的字典，从头建立。")
    subword_encoder_zh = tfds.features.text.SubwordTextEncoder.build_from_corpus(
        (zh.numpy() for _, zh in train_examples),
        target_vocab_size=10000,  # 有需要可以调整整字典大小
        max_subword_length=1)  # 每一个中文字就是字典里的一个单位

    # 將字典檔案存下以方便下次 warmstart
    subword_encoder_zh.save_to_file(zh_vocab_file)

print(f"字典大小：{subword_encoder_zh.vocab_size}")
print(f"前 10 个 subwords：{subword_encoder_zh.subwords[:10]}")
print()


def encode(en_t, zh_t):
    # 因为字典的索引从 0 开始，
    # 我们可以使用 subword_encoder_en.vocab_size 这个值作为 BOS 的索引值
    # 用 subword_encoder_en.vocab_size + 1 作为 EOS 的索引值
    en_indices = [subword_encoder_en.vocab_size] + subword_encoder_en.encode(
        en_t.numpy()) + [subword_encoder_en.vocab_size + 1]
    # 同理，不过是使用中文字典的最后一个索引 + 1
    zh_indices = [subword_encoder_zh.vocab_size] + subword_encoder_zh.encode(
        zh_t.numpy()) + [subword_encoder_zh.vocab_size + 1]

    return en_indices, zh_indices


def tf_encode(en_t, zh_t):
    # 在 `tf_encode` 函式里头的 `en_t` 与 `zh_t` 都不是 Eager Tensors
    # 要到 `tf.py_funtion` 里头才是
    # 另外因为索引都是整数，所以使用 `tf.int64`
    return tf.py_function(encode, [en_t, zh_t], [tf.int64, tf.int64])


MAX_LENGTH = 40


def filter_max_length(en, zh, max_length=MAX_LENGTH):
    # en, zh 分别代表英文与中文的索引序列
    return tf.logical_and(tf.size(en) <= max_length,
                          tf.size(zh) <= max_length)


MAX_LENGTH = 40
BATCH_SIZE = 128
BUFFER_SIZE = 15000

# 训练集
train_dataset = (train_examples  # 输出：(英文句子, 中文句子)
                 .map(tf_encode)  # 输出：(英文索引序列, 中文索引序列)
                 .filter(filter_max_length)  # 同上，且序列长度都不超过 40
                 .shuffle(BUFFER_SIZE)  # 将例子洗牌确保随机性
                 .padded_batch(BATCH_SIZE,  # 将 batch 里的序列都 pad 到一样长度
                               padded_shapes=([-1], [-1]))
                 .prefetch(tf.data.experimental.AUTOTUNE))  # 加速
# 验证集
val_dataset = (val_examples
               .map(tf_encode)
               .filter(filter_max_length)
               .padded_batch(BATCH_SIZE,
                             padded_shapes=([-1], [-1])))

en_batch, zh_batch = next(iter(train_dataset))
print("英文索引序列的 batch")
print(en_batch)
print('-' * 20)
print("中文索引序列的 batch")
print(zh_batch)


def scaled_dot_product_attention(q, k, v, mask):
    """Calculate the attention weights.
    q, k, v must have matching leading dimensions.
    k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
    The mask has different shapes depending on its type(padding or look ahead)
    but it must be broadcastable for addition.

    Args:
      q: query shape == (..., seq_len_q, depth)
      k: key shape == (..., seq_len_k, depth)
      v: value shape == (..., seq_len_v, depth_v)
      mask: Float tensor with shape broadcastable
            to (..., seq_len_q, seq_len_k). Defaults to None.

    Returns:
      output, attention_weights
    """
    # 将 `q`、 `k` 做点积再 scale
    matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)

    dk = tf.cast(tf.shape(k)[-1], tf.float32)  # 取得 seq_k 的序列长度
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)  # scale by sqrt(dk)

    # 将遮罩「加」到被丢入 softmax 前的 logits
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)

    # 取 softmax 是为了得到总和为 1 的比例之后对 `v` 做加权平均
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)

    # 以注意权重对 v 做加权平均（weighted average）
    output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

    return output, attention_weights


# 实作一个执行多头注意力机制的 keras layer
# 在初始的时候指定输出维度 `d_model` & `num_heads，
# 在呼叫的时候输入 `v`, `k`, `q` 以及 `mask`
# 输出跟 scaled_dot_product_attention 函式一样有两个：
# output.shape == (batch_size, seq_len_q, d_model)
# attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
class MultiHeadAttention(tf.keras.layers.Layer):
    # 在初始的时候建立一些必要参数
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads  # 指定要将 `d_model` 拆成几个 heads
        self.d_model = d_model  # 在 split_heads 之前的基底维度

        assert d_model % self.num_heads == 0  # 前面看过，要确保可以平分

        self.depth = d_model // self.num_heads  # 每个 head 里子词的新的 repr. 维度

        self.wq = tf.keras.layers.Dense(d_model)  # 分别给 q, k, v 的 3 个线性转换
        self.wk = tf.keras.layers.Dense(d_model)  # 注意我们并没有指定 activation func
        self.wv = tf.keras.layers.Dense(d_model)

        self.dense = tf.keras.layers.Dense(d_model)  # 多 heads 串接后通过的线性转换

    # 这跟我们前面看过的函式有 87% 相似
    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    # multi-head attention 的实际执行流程，注意参数顺序（这边跟论文以及 TensorFlow 官方教学一致）
    def call(self, v, k, q, mask):
        batch_size = tf.shape(q)[0]

        # 将输入的 q, k, v 都各自做一次线性转换到 `d_model` 维空间
        q = self.wq(q)  # (batch_size, seq_len, d_model)
        k = self.wk(k)  # (batch_size, seq_len, d_model)
        v = self.wv(v)  # (batch_size, seq_len, d_model)

        # 前面看过的，将最后一个 `d_model` 维度分成 `num_heads` 个 `depth` 维度
        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

        # 利用 broadcasting 让每个句子的每个 head 的 qi, ki, vi 都各自进行注意力机制
        # 输出会多一个 head 维度
        scaled_attention, attention_weights = scaled_dot_product_attention(
            q, k, v, mask)
        # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)

        # 跟我们在 `split_heads` 函式做的事情刚好相反，先做 transpose 再做 reshape
        # 将 `num_heads` 个 `depth` 维度串接回原来的 `d_model` 维度
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])
        # (batch_size, seq_len_q, num_heads, depth)
        concat_attention = tf.reshape(scaled_attention,
                                      (batch_size, -1, self.d_model))
        # (batch_size, seq_len_q, d_model)

        # 通过最后一个线性转换
        output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)
        return output, attention_weights


def point_wise_feed_forward_network(d_model, dff):
    # 此 FFN 对输入做两个线性转换，中间加了一个 ReLU activation func
    return tf.keras.Sequential([
        tf.keras.layers.Dense(dff, activation='relu'),  # (batch_size, seq_len, dff)
        tf.keras.layers.Dense(d_model)  # (batch_size, seq_len, d_model)
    ])


# Encoder 里头会有 N 个 EncoderLayers，而每个 EncoderLayer 里又有两个 sub-layers: MHA & FFN
class EncoderLayer(tf.keras.layers.Layer):
    # Transformer 论文内预设 dropout rate 为 0.1
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(EncoderLayer, self).__init__()

        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = point_wise_feed_forward_network(d_model, dff)

        # layer norm 很常在 RNN-based 的模型被使用。一个 sub-layer 一个 layer norm
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        # 一样，一个 sub-layer 一个 dropout layer
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    # 需要丢入 `training` 参数是因为 dropout 在训练以及测试的行为有所不同
    def call(self, x, training, mask):
        # 除了 `attn`，其他张量的 shape 皆为 (batch_size, input_seq_len, d_model)
        # attn.shape == (batch_size, num_heads, input_seq_len, input_seq_len)

        # sub-layer 1: MHA
        # Encoder 利用注意机制关注自己当前的序列，因此 v, k, q 全部都是自己
        # 另外别忘了我们还需要 padding mask 来遮住输入序列中的 <pad> token
        attn_output, attn = self.mha(x, x, x, mask)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)

        # sub-layer 2: FFN
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)  # 记得 training
        out2 = self.layernorm2(out1 + ffn_output)

        return out2


# Decoder 里头会有 N 个 DecoderLayer，
# 而 DecoderLayer 又有三个 sub-layers: 自注意的 MHA, 关注 Encoder 输出的 MHA & FFN
class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(DecoderLayer, self).__init__()

        # 3 个 sub-layers 的主角们
        self.mha1 = MultiHeadAttention(d_model, num_heads)
        self.mha2 = MultiHeadAttention(d_model, num_heads)
        self.ffn = point_wise_feed_forward_network(d_model, dff)

        # 定义每个 sub-layer 用的 LayerNorm
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        # 定义每个 sub-layer 用的 Dropout
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
        self.dropout3 = tf.keras.layers.Dropout(rate)

    def call(self, x, enc_output, training,
             combined_mask, inp_padding_mask):
        # 所有 sub-layers 的主要输出皆为 (batch_size, target_seq_len, d_model)
        # enc_output 为 Encoder 输出序列，shape 为 (batch_size, input_seq_len, d_model)
        # attn_weights_block_1 则为 (batch_size, num_heads, target_seq_len, target_seq_len)
        # attn_weights_block_2 则为 (batch_size, num_heads, target_seq_len, input_seq_len)

        # sub-layer 1: Decoder layer 自己对输出序列做注意力。
        # 我们同时需要 look ahead mask 以及输出序列的 padding mask
        # 来避免前面已生成的子词关注到未来的子词以及 <pad>
        attn1, attn_weights_block1 = self.mha1(x, x, x, combined_mask)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(attn1 + x)

        # sub-layer 2: Decoder layer 关注 Encoder 的最后输出
        # 记得我们一样需要对 Encoder 的输出套用 padding mask 避免关注到 <pad>
        attn2, attn_weights_block2 = self.mha2(
            enc_output, enc_output, out1, inp_padding_mask)  # (batch_size, target_seq_len, d_model)
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layernorm2(attn2 + out1)  # (batch_size, target_seq_len, d_model)

        # sub-layer 3: FFN 部分跟 Encoder layer 完全一样
        ffn_output = self.ffn(out2)  # (batch_size, target_seq_len, d_model)

        ffn_output = self.dropout3(ffn_output, training=training)
        out3 = self.layernorm3(ffn_output + out2)  # (batch_size, target_seq_len, d_model)

        # 除了主要输出 `out3` 以外，输出 multi-head 注意权重方便之后理解模型内部状况
        return out3, attn_weights_block1, attn_weights_block2


# 以下直接參考 TensorFlow 官方 tutorial
def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
    return pos * angle_rates


def positional_encoding(position, d_model):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                            np.arange(d_model)[np.newaxis, :],
                            d_model)

    # apply sin to even indices in the array; 2i
    sines = np.sin(angle_rads[:, 0::2])

    # apply cos to odd indices in the array; 2i+1
    cosines = np.cos(angle_rads[:, 1::2])

    pos_encoding = np.concatenate([sines, cosines], axis=-1)

    pos_encoding = pos_encoding[np.newaxis, ...]

    return tf.cast(pos_encoding, dtype=tf.float32)


seq_len = 50
d_model = 512

pos_encoding = positional_encoding(seq_len, d_model)
pos_encoding


class Encoder(tf.keras.layers.Layer):
    # Encoder 的初始参数除了本来就要给 EncoderLayer 的参数还多了：
    # - num_layers: 决定要有几个 EncoderLayers, 前面影片中的 `N`
    # - input_vocab_size: 用来把索引转成词嵌入向量
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size,
                 rate=0.1):
        super(Encoder, self).__init__()

        self.d_model = d_model

        self.embedding = tf.keras.layers.Embedding(input_vocab_size, d_model)
        self.pos_encoding = positional_encoding(input_vocab_size, self.d_model)

        # 建立 `num_layers` 个 EncoderLayers
        self.enc_layers = [EncoderLayer(d_model, num_heads, dff, rate)
                           for _ in range(num_layers)]

        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mask):
        # 输入的 x.shape == (batch_size, input_seq_len)
        # 以下各 layer 的输出皆为 (batch_size, input_seq_len, d_model)
        input_seq_len = tf.shape(x)[1]

        # 将 2 维的索引序列转成 3 维的词嵌入张量，并依照论文乘上 sqrt(d_model)
        # 再加上对应长度的位置编码
        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :input_seq_len, :]

        # 对 embedding 跟位置编码的总合做 regularization
        # 这在 Decoder 也会做
        x = self.dropout(x, training=training)

        # 通过 N 个 EncoderLayer 做编码
        for i, enc_layer in enumerate(self.enc_layers):
            x = enc_layer(x, training, mask)
            # 以下只是用来 demo EncoderLayer outputs
            # print('-' * 20)
            # print(f"EncoderLayer {i + 1}'s output:", x)

        return x


class Decoder(tf.keras.layers.Layer):
    # 初始参数跟 Encoder 只差在用 `target_vocab_size` 而非 `inp_vocab_size`
    def __init__(self, num_layers, d_model, num_heads, dff, target_vocab_size,
                 rate=0.1):
        super(Decoder, self).__init__()

        self.d_model = d_model

        # 为中文（目标语言）建立词嵌入层
        self.embedding = tf.keras.layers.Embedding(target_vocab_size, d_model)
        self.pos_encoding = positional_encoding(target_vocab_size, self.d_model)

        self.dec_layers = [DecoderLayer(d_model, num_heads, dff, rate)
                           for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(rate)

    # 呼叫时的参数跟 DecoderLayer 一模一样
    def call(self, x, enc_output, training,
             combined_mask, inp_padding_mask):
        tar_seq_len = tf.shape(x)[1]
        attention_weights = {}  # 用来存放每个 Decoder layer 的注意权重

        # 这边跟 Encoder 做的事情完全一样
        x = self.embedding(x)  # (batch_size, tar_seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :tar_seq_len, :]
        x = self.dropout(x, training=training)

        for i, dec_layer in enumerate(self.dec_layers):
            x, block1, block2 = dec_layer(x, enc_output, training,
                                          combined_mask, inp_padding_mask)

            # 将从每个 Decoder layer 取得的注意权重全部存下来回传，方便我们观察
            attention_weights['decoder_layer{}_block1'.format(i + 1)] = block1
            attention_weights['decoder_layer{}_block2'.format(i + 1)] = block2

        # x.shape == (batch_size, tar_seq_len, d_model)
        return x, attention_weights


# Transformer 之上已经没有其他 layers 了，我们使用 tf.keras.Model 建立一个模型
class Transformer(tf.keras.Model):
    # 初始参数包含 Encoder & Decoder 都需要超参数以及中英字典数目
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size,
                 target_vocab_size, rate=0.1):
        super(Transformer, self).__init__()

        self.encoder = Encoder(num_layers, d_model, num_heads, dff,
                               input_vocab_size, rate)

        self.decoder = Decoder(num_layers, d_model, num_heads, dff,
                               target_vocab_size, rate)
        # 这个 FFN 输出跟中文字典一样大的 logits 数，等通过 softmax 就代表每个中文字的出现机率
        self.final_layer = tf.keras.layers.Dense(target_vocab_size)

    # enc_padding_mask 跟 dec_padding_mask 都是英文序列的 padding mask，
    # 只是一个给 Encoder layer 的 MHA 用，一个是给 Decoder layer 的 MHA 2 使用
    def call(self, inp, tar, training, enc_padding_mask,
             combined_mask, dec_padding_mask):
        enc_output = self.encoder(inp, training, enc_padding_mask)  # (batch_size, inp_seq_len, d_model)

        # dec_output.shape == (batch_size, tar_seq_len, d_model)
        dec_output, attention_weights = self.decoder(
            tar, enc_output, training, combined_mask, dec_padding_mask)

        # 将 Decoder 输出通过最后一个 linear layer
        final_output = self.final_layer(dec_output)  # (batch_size, tar_seq_len, target_vocab_size)

        return final_output, attention_weights


loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction='none')


def loss_function(real, pred):
    # 这次的 mask 将序列中不等于 0 的位置视为 1，其余为 0
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    # 照样计算所有位置的 cross entropy 但不加总
    loss_ = loss_object(real, pred)
    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask  # 只计算非 <pad> 位置的损失

    return tf.reduce_mean(loss_)


train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
    name='train_accuracy')

num_layers = 4
d_model = 128
dff = 512
num_heads = 8

input_vocab_size = subword_encoder_en.vocab_size + 2
target_vocab_size = subword_encoder_zh.vocab_size + 2
dropout_rate = 0.1  # 预设值

print("input_vocab_size:", input_vocab_size)
print("target_vocab_size:", target_vocab_size)


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    # 论文预设 `warmup_steps` = 4000
    def __init__(self, d_model, warmup_steps=4000):
        super(CustomSchedule, self).__init__()

        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)

        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)


# 將定制化 learning rate schdeule 丟入 Adam opt.
# Adam opt. 的参数和论文相同
learning_rate = CustomSchedule(d_model)
optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98,
                                     epsilon=1e-9)

transformer = Transformer(num_layers, d_model, num_heads, dff,
                          input_vocab_size, target_vocab_size, dropout_rate)

print(f"""这个 Transformer 有 {num_layers} 层 Encoder / Decoder layers
d_model: {d_model}
num_heads: {num_heads}
dff: {dff}
input_vocab_size: {input_vocab_size}
target_vocab_size: {target_vocab_size}
dropout_rate: {dropout_rate}
""")

# 方便比较不同实验/ 不同超参数设定的结果
run_id = f"{num_layers}layers_{d_model}d_{num_heads}heads_{dff}dff"
checkpoint_path = os.path.join(checkpoint_path, run_id)
log_dir = os.path.join(log_dir, run_id)

# tf.train.Checkpoint 可以帮我们把想要存下来的东西整合起来，方便储存与读取
# 一般来说你会想存下模型以及 optimizer 的状态
ckpt = tf.train.Checkpoint(transformer=transformer,
                           optimizer=optimizer)

# ckpt_manager 会去 checkpoint_path 看有没有符合 ckpt 里头定义的东西
# 存档的时候只保留最近 5 次 checkpoints，其他自动删除
ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

# 如果在 checkpoint 路径上有发现档案就读进来
if ckpt_manager.latest_checkpoint:
    ckpt.restore(ckpt_manager.latest_checkpoint)

    # 用来确认之前训练多少 epochs 了
    last_epoch = int(ckpt_manager.latest_checkpoint.split("-")[-1])
    print(f'已读取最新的 checkpoint，模型已训练 {last_epoch} epochs。')
else:
    last_epoch = 0
    print("没找到 checkpoint，从头训练。")


# 为 Transformer 的 Encoder / Decoder 准备遮罩


def create_padding_mask(seq):
    # padding mask 的工作就是把索引序列中为 0 的位置设为 1
    mask = tf.cast(tf.equal(seq, 0), tf.float32)
    return mask[:, tf.newaxis, tf.newaxis, :]  # broadcasting


# 其遮罩为一个右上角的三角形
def create_look_ahead_mask(size):
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask  # (seq_len, seq_len)


def create_masks(inp, tar):
    # 英文句子的 padding mask，要交给 Encoder layer 自注意力机制用的
    enc_padding_mask = create_padding_mask(inp)

    # 同样也是英文句子的 padding mask，但是是要交给 Decoder layer 的 MHA 2
    # 关注 Encoder 输出序列用的
    dec_padding_mask = create_padding_mask(inp)

    # Decoder layer 的 MHA1 在做自注意力机制用的
    # `combined_mask` 是中文句子的 padding mask 跟 look ahead mask 的叠加
    look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])
    dec_target_padding_mask = create_padding_mask(tar)
    combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)

    return enc_padding_mask, combined_mask, dec_padding_mask


@tf.function  # 让 TensorFlow 帮我们将 eager code 优化并加快运算
def train_step(inp, tar):
    # 前面说过的，用去尾的原始序列去预测下一个字的序列
    tar_inp = tar[:, :-1]
    tar_real = tar[:, 1:]

    # 建立 3 个遮罩
    enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inp, tar_inp)

    # 纪录 Transformer 的所有运算过程以方便之后做梯度下降
    with tf.GradientTape() as tape:
        # 注意是丢入 `tar_inp` 而非 `tar`。记得将 `training` 参数设定为 True
        predictions, _ = transformer(inp, tar_inp,
                                     True,
                                     enc_padding_mask,
                                     combined_mask,
                                     dec_padding_mask)
        # 跟影片中显示的相同，计算左移一个字的序列跟模型预测分布之间的差异，当作 loss
        loss = loss_function(tar_real, predictions)

    # 取出梯度并呼叫前面定义的 Adam optimizer 帮我们更新 Transformer 里头可训练的参数
    gradients = tape.gradient(loss, transformer.trainable_variables)
    optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))

    # 将 loss 以及训练 acc 记录到 TensorBoard 上，非必要
    train_loss(loss)
    train_accuracy(tar_real, predictions)


# 定义我们要看几遍数据集
EPOCHS = 30
print(f"此超参数组合的 Transformer 已经训练 {last_epoch} epochs。")
print(f"剩余 epochs：{min(0, last_epoch - EPOCHS)}")

# 用来写资讯到 TensorBoard，非必要但十分推荐
summary_writer = tf.summary.create_file_writer(log_dir)

# 比对设定的 `EPOCHS` 以及已训练的 `last_epoch` 来决定还要训练多少 epochs
for epoch in range(last_epoch, EPOCHS):
    start = time.time()

    # 重置纪录 TensorBoard 的 metrics
    train_loss.reset_states()
    train_accuracy.reset_states()

    # 一个 epoch 就是把我们定义的训练资料集一个一个 batch 拿出来处理，直到看完整个数据集
    for (step_idx, (inp, tar)) in enumerate(train_dataset):
        # 每次 step 就是将数据丢入 Transformer，让它生预测结果并计算梯度最小化 loss
        train_step(inp, tar)

    # 每个 epoch 完成就存一次档
    if (epoch + 1) % 1 == 0:
        ckpt_save_path = ckpt_manager.save()
        print('Saving checkpoint for epoch {} at {}'.format(epoch + 1,
                                                            ckpt_save_path))

    # 将 loss 以及 accuracy 写到 TensorBoard 上
    with summary_writer.as_default():
        tf.summary.scalar("train_loss", train_loss.result(), step=epoch + 1)
        tf.summary.scalar("train_acc", train_accuracy.result(), step=epoch + 1)

    print('Epoch {} Loss {:.4f} Accuracy {:.4f}'.format(epoch + 1,
                                                        train_loss.result(),
                                                        train_accuracy.result()))
    print('Time taken for 1 epoch: {} secs\n'.format(time.time() - start))

# 给定一个英文句子，输出预测的中文索引数字序列以及注意权重 dict
def evaluate(inp_sentence):
    # 准备英文句子前后会加上的 <start>, <end>
    start_token = [subword_encoder_en.vocab_size]
    end_token = [subword_encoder_en.vocab_size + 1]

    # inp_sentence 是字串，我们用 Subword Tokenizer 将其变成子词的索引序列
    # 并在前后加上 BOS / EOS
    inp_sentence = start_token + subword_encoder_en.encode(inp_sentence) + end_token
    encoder_input = tf.expand_dims(inp_sentence, 0) #[batch,seq]

    # 跟我们在影片里看到的一样，Decoder 在第一个时间点吃进去的输入
    # 是一个只包含一个中文 <start> token 的序列
    decoder_input = [subword_encoder_zh.vocab_size] #'《start》'
    output = tf.expand_dims(decoder_input, 0)  # 增加 batch 维度

    # auto-regressive，一次生成一个中文字并将预测加到输入再度喂进 Transformer
    for i in range(MAX_LENGTH):
        # 每多一个生成的字就得产生新的遮罩
        enc_padding_mask, combined_mask, dec_padding_mask = create_masks(
            encoder_input, output)

        # predictions.shape == (batch_size, seq_len, vocab_size)
        predictions, attention_weights = transformer(encoder_input,
                                                     output,
                                                     False,
                                                     enc_padding_mask,
                                                     combined_mask,
                                                     dec_padding_mask)

        # 将序列中最后一个 distribution 取出，并将里头值最大的当作模型最新的预测字
        predictions = predictions[:, -1:, :]  # (batch_size, 1, vocab_size)

        predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)

        # 遇到 <end> token 就停止回传，代表模型已经产生完结果
        if tf.equal(predicted_id, subword_encoder_zh.vocab_size + 1):
            return tf.squeeze(output, axis=0), attention_weights

        # 将 Transformer 新预测的中文索引加到输出序列中，让 Decoder 可以在产生
        # 下个中文字的时候关注到最新的 `predicted_id`
        output = tf.concat([output, predicted_id], axis=-1)

    # 将 batch 的维度去掉后回传预测的中文索引序列
    return tf.squeeze(output, axis=0), attention_weights





# 要被翻译的英文句子
sentence = "China has enjoyed continuing economic growth."

# 取得预测的中文索引序列
predicted_seq, _ = evaluate(sentence)

# 过滤掉 <start> & <end> tokens 并用中文的 subword tokenizer 帮我们将索引序列还原回中文句子
target_vocab_size = subword_encoder_zh.vocab_size
predicted_seq_without_bos_eos = [idx for idx in predicted_seq if idx < target_vocab_size]
predicted_sentence = subword_encoder_zh.decode(predicted_seq_without_bos_eos)

print("sentence:", sentence)
print("-" * 20)
print("predicted_seq:", predicted_seq)
print("-" * 20)
print("predicted_sentence:", predicted_sentence)