import  sys
import os


import tensorflow as tf
from model import esim
from utils.load_data import load_char_data

p, h, y = load_char_data('./data/atec_nlp_sim_train.csv', data_size=None)
p_eval, h_eval, y_eval = load_char_data('./data/atec_nlp_sim_train_add.csv', data_size=1000)

print(p.shape,h.shape,y.shape)

# gen dataset
train_ds = tf.data.Dataset.from_tensor_slices(((p, h),y))
train_ds = train_ds.batch(16)
# for i in train_ds:
#     print(i)
#     break
#定义损失函数、评估函数、优化器
learning_rate = 0.001

loss_object = tf.keras.losses.SparseCategoricalCrossentropy()


optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)


train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')


model = esim( rate=0.1, num_labels=2)
#定义train_step
def train_one_step(question1,question2, labels):
    with tf.GradientTape() as tape:
        predictions = model((question1,question2))
        loss = loss_object(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss(loss) #update
    train_accuracy(labels, predictions)#update



#定义eval_step
def test_one_step(contents, labels):
    predictions = model(contents)
    t_loss = loss_object(labels, predictions)

    test_loss(t_loss)
    test_accuracy(labels, predictions)



EPOCHS = 10
for epoch in range(EPOCHS):
    # 在下一个epoch开始时，重置评估指标
    train_loss.reset_states()
    train_accuracy.reset_states()
    test_loss.reset_states()
    test_accuracy.reset_states()


    for step, ((x1,x2), labels) in enumerate(train_ds):
        train_one_step(x1,x2, labels)

        if step % 200 == 0:
            print("step:{0}; Samples:{1}; Train Loss:{2}; Train Accuracy:{3}".format(step,
                                                                                     ( step + 1) * 1024,
                                                                                     train_loss.result(),
                                                                                     train_accuracy.result() * 100
                                                                                     ))

    template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
    print(template.format(epoch + 1,
                          train_loss.result(),
                          train_accuracy.result() * 100,
                          test_loss.result(),
                          test_accuracy.result() * 100,
                          ))