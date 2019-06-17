#!/usr/bin/env python
# encoding: utf-8
'''
@author: tianxiaomo
@license: (C) Apache.
@contact: huguanghao520@gmail.com
@software: PyCharm
@file: demo.py
@time: 2019/6/3 22:07
@desc:
'''
import tensorflow as tf
from keras.layers import *
from keras.models import Model
import keras.backend as K
from keras.callbacks import Callback
from keras.optimizers import Nadam

import json
import random
from random import choice
from tqdm import tqdm
from sklearn.model_selection import train_test_split

from py_rc import log

np.random.seed(2019)
random.seed(2019)
tf.set_random_seed(2019)
# from multipliers import M_Nadam

def gpu_set(gpu_num):
    import tensorflow as tf
    import keras.backend.tensorflow_backend as KTF
    import os
    if isinstance(gpu_num, (list, tuple)):
        gpu_num = ','.join(str(i) for i in gpu_num)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_num)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    sess = tf.Session(config=config)
    KTF.set_session(sess)
    print('GPU config done!')

gpu_set(3)

data = json.load(open('small_train_data.json',encoding='utf-8')).get('data')
id2char, char2id = json.load(open('all_chars.json',encoding='utf-8'))
train_data,dev_data = train_test_split(np.arange(len(data)),test_size=0.2)
train_data = list(np.asarray(data)[train_data])
dev_data = list(np.asarray(data)[dev_data])

char_size = 128

logger = log.init_logger(log_path='logs')
logger.info('subject+2cnn(dilation_rate=4)+object_cnn(dilation_rate=2)')

def seq_padding(X):
    L = [len(x) for x in X]
    ML = max(L)
    return [x + [0] * (ML - len(x)) for x in X]

def seq_gather(x):
    """seq是[None, seq_len, s_size]的格式，
    idxs是[None, 1]的格式，在seq的第i个序列中选出第idxs[i]个向量，
    最终输出[None, s_size]的向量。
    """
    seq, idxs = x
    idxs = K.cast(idxs, 'int32')
    batch_idxs = K.arange(0, K.shape(seq)[0])
    batch_idxs = K.expand_dims(batch_idxs, 1)
    idxs = K.concatenate([batch_idxs, idxs], 1)
    return K.tf.gather_nd(seq, idxs)

def seq_gather1(x):
    """seq是[None, seq_len, s_size]的格式，
    idxs是[None, 1]的格式，在seq的第i个序列中选出第idxs[i]个向量，
    最终输出[None, s_size]的向量。
    """
    seq, idxs1,idxs2 = x
    idxs1 = K.cast(idxs1, 'int32')
    idxs2 = K.cast(idxs2, 'int32')
    batch_idxs = K.arange(0, K.shape(seq)[0])
    batch_idxs = K.expand_dims(batch_idxs, 1)
    idxs = K.concatenate([batch_idxs, idxs1], 1)
    return K.tf.gather_nd(seq, idxs)

def seq_and_vec(x):
    """seq是[None, seq_len, s_size]的格式，
    vec是[None, v_size]的格式，将vec重复seq_len次，拼到seq上，
    得到[None, seq_len, s_size+v_size]的向量。
    """
    seq, vec = x
    vec = K.expand_dims(vec, 1)
    vec = K.zeros_like(seq[:, :, :1]) + vec
    return K.concatenate([seq, vec], 2)

def seq_maxpool(x):
    """seq是[None, seq_len, s_size]的格式，
    mask是[None, seq_len, 1]的格式，先除去mask部分，
    然后再做maxpooling。
    """
    seq, mask_t = x
    seq -= (1 - mask_t) * 1e10
    return K.max(seq, 1)


class data_generator:
    def __init__(self, data, batch_size=16):
        # self.data = data.get('data')
        self.data = data
        self.batch_size = batch_size
        self.steps = len(self.data) // self.batch_size
        if len(self.data) % self.batch_size != 0:
            self.steps += 1
    def __len__(self):
        return self.steps
    def __iter__(self):
        while True:
            idxs = list(range(len(self.data)))
            np.random.shuffle(idxs)
            T,A1,A2,Q,QA,AN = [],[],[],[],[],[] #文本,问题,答案位置1,答案位置2,解决回答,是,否
            for i in idxs:
                d = self.data[i]['paragraphs'][0]
                text = d['context']
                sp = choice(d['qas'])
                q = sp['question']
                if len(sp['answers']) == 0:                 # 不回答
                    a1, a2 = [0] * len(text), [0] * len(text)
                    an = [0,1,0,0]
                elif sp['answers'][0]['answer_start'] != -1:    # 回答
                    an1 = sp['answers'][0]['answer_start']
                    an2 = an1 + len(sp['answers'][0]['text'])

                    a1, a2 = [0] * len(text), [0] * len(text)
                    a1[an1] = 1
                    a2[an2 - 1] = 1
                    an = [1,0,0,0]
                elif sp['answers'][0]['text'] == 'YES':         # yes
                    a1, a2 = [0] * len(text), [0] * len(text)
                    an = [0, 0, 1, 0]
                else:                                           # yes
                    a1, a2 = [0] * len(text), [0] * len(text)
                    an = [0, 0, 0, 1]

                T.append([char2id.get(c, 1) for c in text])  # 1是unk，0是padding
                Q.append([char2id.get(c, 1) for c in q])
                A1.append(a1)
                A2.append(a2)
                AN.append(an)

                if len(T) == self.batch_size or i == idxs[-1]:
                    T = np.array(seq_padding(T))
                    Q = np.array(seq_padding(Q))
                    A1 = np.array(seq_padding(A1))
                    A2 = np.array(seq_padding(A2))
                    AN = np.array(AN)
                    yield [T,Q,A1,A2,AN], None
                    T,Q,A1,A2,AN = [],[],[],[],[]


# input
t_in = Input(shape=(None,))
q_in = Input(shape=(None,))
a1_in = Input(shape=(None,))
a2_in = Input(shape=(None,))
an_in = Input(shape=(4,))

t, q, a1, a2,a = t_in, q_in, a1_in, a2_in, an_in

#question encoder
mask_q = Lambda(lambda x: K.cast(K.greater(K.expand_dims(x, 2), 0), 'float32'))(q)
q = Embedding(len(char2id)+2, char_size,name='question_enbeding')(q) # 0: padding, 1: unk
q = Dropout(0.25)(q)
q = Lambda(lambda x: x[0] * x[1])([q, mask_q])
q = Bidirectional(LSTM(char_size//2, return_sequences=True),name='q_blstm_1')(q)
q = Bidirectional(LSTM(char_size//2, return_sequences=True),name='q_blstm_2')(q)
q_max = Lambda(seq_maxpool)([q, mask_q])

# text encoder
mask_t = Lambda(lambda x: K.cast(K.greater(K.expand_dims(x, 2), 0), 'float32'))(t)
t = Embedding(len(char2id)+2, char_size,name='text_enbeding')(t) # 0: padding, 1: unk
t = Dropout(0.25)(t)
t = Lambda(lambda x: x[0] * x[1])([t, mask_t])
t = Bidirectional(LSTM(char_size//2, return_sequences=True),name='t_blstm_1')(t)
t = Bidirectional(LSTM(char_size//2, return_sequences=True),name='t_blstm_2')(t)
# t = Conv1D(char_size, 3,activation='relu' ,padding='same',name='encoder_cnn1')(t)
# t = Conv1D(char_size, 3,dilation_rate=2 ,activation='relu' ,padding='same',name='encoder_cnn2')(t)
# t = Conv1D(char_size, 3,dilation_rate=4 ,activation='relu' ,padding='same',name='encoder_cnn3')(t)

# t = Dropout(0.2)(t)
t_max = Lambda(seq_maxpool)([t, mask_t])
t_dim = K.int_shape(t)[-1]

#attention
h = Lambda(seq_and_vec, output_shape=(None, t_dim*2))([t, t_max])
h = Lambda(seq_and_vec, output_shape=(None, t_dim*3))([h, q_max])
#decoder
h = Conv1D(char_size, 3, activation='relu', padding='same',name='obj_decoder_cnn1')(h)
h = Conv1D(char_size, 3, dilation_rate=2,activation='relu', padding='same',name='obj_decoder_cnn2')(h)
h = Conv1D(char_size, 3, dilation_rate=4,activation='relu', padding='same',name='obj_decoder_cnn3')(h)
an = Conv1D(4,3,activation='sigmoid',name='obj_decoder_fc_po')(h)
an = GlobalAveragePooling1D()(an)
an1 = Dense(1, activation='sigmoid',name='obj_decoder_fc_po1')(h)
an2 = Dense(1, activation='sigmoid',name='obj_decoder_fc_po2')(h)

#model
train_model = Model([t_in, q_in, a1_in, a2_in, an_in],[an1,an2,an])
test_model = Model([t_in,q_in],[an1,an2,an])

# loss
a1 = K.expand_dims(a1, 2)
a2 = K.expand_dims(a2, 2)

s1_loss = K.binary_crossentropy(a1, an1)
s1_loss = K.sum(s1_loss * mask_t) / K.sum(mask_t)
s2_loss = K.binary_crossentropy(a2, an2)
s2_loss = K.sum(s2_loss * mask_t) / K.sum(mask_t)
an_loss = K.categorical_crossentropy(a,an)

loss = an_loss + s1_loss + s2_loss

train_model.add_loss(loss)

lr_dict = {
    'word_enbeding': 1,
    'blstm_1': 1,
    'blstm_2': 1,

    'sub_decoder_cnn1': 1,
    'sub_decoder_cnn2': 1,
    'sub_decoder_fc_ps1': 1,
    'sub_decoder_fc_ps2': 1,

    'obj_decoder_cnn1': 1,
    'obj_decoder_cnn2':1,
    'obj_decoder_fc_po1':1,
    'obj_decoder_fc_po2':1
}

train_model.compile(optimizer=Nadam(0.001))
train_model.summary(print_fn=logger.info)
train_model.load_weights('best_m_model.weights')

def anwsers(text_in,question_in):
    R = []
    _t = [char2id.get(c, 1) for c in text_in]
    _q = [char2id.get(c, 1) for c in question_in]
    _t = np.array([_t])
    _q = np.array([_q])
    _a1, _a2,_an = test_model.predict([_t,_q])
    _an = _an[0]
    _a1, _a2 = _a1[0, :, 0], _a2[0, :, 0]
    return _a1,_a2,_an


class Evaluate(Callback):

    def __init__(self,patience):
        super(Evaluate,self).__init__()
        self.F1 = []
        self.best = 0.
        self.wait = 0
        self.patience = patience
    def on_epoch_end(self, epoch, logs=None):
        acc = self.evaluate()
        self.F1.append(acc)
        if acc > self.best:
            self.best = acc
            train_model.save_weights('best_m_model.weights')
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.model.stop_training = True

        logger.info('Epoch: %d,  loss: %.5f,acc: %.4f ---- best acc: %.4f' % (epoch, logs.get('loss').mean(),acc, self.best))

    def evaluate(self):
        num = 0
        # _data = dev_data.get('data')
        _data = dev_data
        for d in tqdm(_data):
            _d = d['paragraphs'][0]
            _t = _d['context']
            sp = choice(_d['qas'])
            _q = sp['question']
            an1,an2,an_c = anwsers(_t,_q)
            an1 = an1.argmax()
            an2 = an2.argmax()
            an_c = an_c.argmax()

            if len(sp['answers']) == 0:  # 不回答
                if an_c == 1:
                    num += 1
            elif sp['answers'][0]['answer_start'] != -1:  # 回答
                a1 = sp['answers'][0]['answer_start']
                a2 = a1 + len(sp['answers'][0]['text'])
                if an_c == 0 and an1 == a1 and an2 ==  a2:
                    num += 1
            elif sp['answers'][0]['text'] == 'YES':  # yes
                if an_c == 2:
                    num += 1
            else:  # no
                if an_c == 3:
                    num += 1

        return num / len(_data)


train_D = data_generator(train_data)
evaluator = Evaluate(patience=5)
train_model.fit_generator(train_D.__iter__(),
                          steps_per_epoch=len(train_D),
                          epochs=10,
                          callbacks=[evaluator]
                          )

#
# lr_dict = {
#     'word_enbeding': 0.1,
#     'blstm_1': 0.01,
#     'blstm_2': 0.01,
#
#     'sub_decoder_cnn1': 0.1,
#     'sub_decoder_cnn2': 0.1,
#     'sub_decoder_fc_ps1': 1,
#     'sub_decoder_fc_ps2': 1,
#
#     'obj_decoder_cnn1': 0.1,
#     'obj_decoder_cnn2':0.1,
#     'obj_decoder_fc_po1':1,
#     'obj_decoder_fc_po2':1
# }
# train_model.load_weights('best_m_model.weights')
# train_model.compile(optimizer=Nadam(0.001,multipliers=lr_dict))
# train_model.fit_generator(train_D.__iter__(),
#                           steps_per_epoch=len(train_D),
#                           epochs=100,
#                           # callbacks=[evaluator]
#                           )