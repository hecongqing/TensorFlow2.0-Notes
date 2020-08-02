import pandas as pd
from preprocess import gen_data_set, gen_model_input
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from model import MIND
import  logging
import os
import time

def sampledsoftmaxloss(y_true, y_pred):
    return tf.keras.backend.mean(y_pred)


data = pd.read_csv("./movielens_sample.txt")
sparse_features = ["movie_id", "user_id",
                   "gender", "age", "occupation", "zip", ]
SEQ_LEN = 50

# 1.Label Encoding for sparse features,and process sequence features with `gen_date_set` and `gen_model_input`

features = ['user_id', 'movie_id', 'gender', 'age', 'occupation', 'zip']
feature_max_idx = {}
for feature in features:
    lbe = LabelEncoder()
    data[feature] = lbe.fit_transform(data[feature]) + 1
    feature_max_idx[feature] = data[feature].max() + 1

user_profile = data[["user_id", "gender", "age", "occupation", "zip"]].drop_duplicates('user_id')

item_profile = data[["movie_id"]].drop_duplicates('movie_id')

user_profile.set_index("user_id", inplace=True)

user_item_list = data.groupby("user_id")['movie_id'].apply(list)

train_set, test_set = gen_data_set(data, 0)

train_model_input, train_label = gen_model_input(train_set, user_profile, SEQ_LEN)
test_model_input, test_label = gen_model_input(test_set, user_profile, SEQ_LEN)


train_dataset = tf.data.Dataset.from_tensor_slices((train_model_input, train_label))
train_dataset =  train_dataset.batch(32)


# 2.count #unique features for each sparse field and generate feature config for sequence feature

embedding_dim = 16

model = MIND(k_max=5, dynamic_k=False, embedding_dim=16, seq_max_len=SEQ_LEN, p=1, num_sampled=10,
             feature_max_idx=feature_max_idx)

learning_rate = 0.002
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

train_loss = tf.keras.metrics.Mean(name='train_loss')


def train_one_step(features, labels):
    with tf.GradientTape() as tape:
        predictions = model(features, training=True)
        loss = sampledsoftmaxloss(labels, predictions)

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    train_loss(loss)

# 方便比较不同实验/ 不同超参数设定的结果
checkpoint_path = os.path.join('./checkpoints/', "MIND_V2")

ckpt = tf.train.Checkpoint(model=model,
                           optimizer=optimizer)

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
    logging.info("没找到 checkpoint，从头训练。")

EPOCHS = 1 - last_epoch
for epoch in range(EPOCHS):

    start = time.time()

    train_loss.reset_states()


    for step, (features, labels) in enumerate(train_dataset):
        train_one_step(features, labels)
    print('Epoch {} Train Loss {:.4f}'.format(epoch + 1,
                                              train_loss.result() ))
    print('Time taken for 1 epoch: {} secs\n'.format(time.time() - start))
    # model.compile(optimizer="adam", loss=sampledsoftmaxloss)
    #
    # history = model.fit(train_model_input, train_label,
    #                     batch_size=256, epochs=1, verbose=1, validation_split=0.0, )
    # print(model.summary())
    # 4. Generate user features for testing and full item features for retrieval
    # test_user_model_input = test_model_input
    # all_item_model_input = {"movie_id": item_profile['movie_id'].values}

    # user_embedding_model = Model(inputs=model.user_input, outputs=model.user_embedding)
    # item_embedding_model = Model(inputs=model.item_input, outputs=model.item_embedding)
    #
    # user_embs = user_embedding_model.predict(test_user_model_input, batch_size=2 ** 12)
    # # user_embs = user_embs[:, i, :]  # i in [0,k_max) if MIND
    # item_embs = item_embedding_model.predict(all_item_model_input, batch_size=2 ** 12)
    #
    # print(user_embs.shape)
    # print(item_embs.shape)

    # 5. [Optional] ANN search by faiss  and evaluate the result

    # test_true_label = {line[0]:[line[2]] for line in test_set}
    #
    # import numpy as np
    # import faiss
    # from tqdm import tqdm
    # from deepmatch.utils import recall_N
    #
    # index = faiss.IndexFlatIP(embedding_dim)
    # # faiss.normalize_L2(item_embs)
    # index.add(item_embs)
    # # faiss.normalize_L2(user_embs)
    # D, I = index.search(np.ascontiguousarray(user_embs), 50)
    # s = []
    # hit = 0
    # for i, uid in tqdm(enumerate(test_user_model_input['user_id'])):
    #     try:
    #         pred = [item_profile['movie_id'].values[x] for x in I[i]]
    #         filter_item = None
    #         recall_score = recall_N(test_true_label[uid], pred, N=50)
    #         s.append(recall_score)
    #         if test_true_label[uid] in pred:
    #             hit += 1
    #     except:
    #         print(i)
    # print("recall", np.mean(s))
    # print("hr", hit / len(test_user_model_input['user_id']))
