#!/usr/bin/env python
# encoding: utf-8
'''
@author: tianxiaomo
@license: (C) Apache.
@contact: huguanghao520@gmail.com
@software: PyCharm
@file: demo1.py
@time: 2019/6/8 16:44
@desc:
'''
from __future__ import print_function
import json
from random import choice
from tqdm import tqdm
# import pyhanlp
import re
from sklearn.model_selection import train_test_split

from keras.layers import *
from keras.models import Model
import keras.backend as K
from keras.callbacks import Callback
from keras.optimizers import Adam

from py_rc import log

mode = 0
char_size = 128
maxlen_q = 128
maxlen_t = 1024
#
# word2vec = Word2Vec.load('word2vec_baike')

# id2word = {i+1:j for i,j in enumerate(word2vec.wv.index2word)}
# word2id = {j:i for i,j in id2word.items()}
# word2vec = word2vec.wv.syn0
# word_size = word2vec.shape[1]
# word2vec = np.concatenate([np.zeros((1, word_size)), word2vec])

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

gpu_set([0,1,2,3])


logger = log.init_logger(log_path='logs')
logger.info('subject+2cnn(dilation_rate=4)+object_cnn(dilation_rate=2)')


# def tokenize(s):
#     return [i.word for i in pyhanlp.HanLP.segment(s)]


# def sent2vec(S):
#     """S格式：[[w1, w2]]
#     """
#     V = []
#     for s in S:
#         V.append([])
#         for w in s:
#             for _ in w:
#                 V[-1].append(word2id.get(w, 0))
#     V = seq_padding(V)
#     V = word2vec[V]
#     return V


def repair(d):
    d['text'] = d['text'].lower()
    something = re.findall(u'《([^《》]*?)》', d['text'])
    something = [s.strip() for s in something]
    zhuanji = []
    gequ = []
    for sp in d['spo_list']:
        sp[0] = sp[0].strip(u'《》').strip().lower()
        sp[2] = sp[2].strip(u'《》').strip().lower()
        for some in something:
            if sp[0] in some and d['text'].count(sp[0]) == 1:
                sp[0] = some
        if sp[1] == u'所属专辑':
            zhuanji.append(sp[2])
            gequ.append(sp[0])
    spo_list = []
    for sp in d['spo_list']:
        if sp[1] in [u'歌手', u'作词', u'作曲']:
            if sp[0] in zhuanji and sp[0] not in gequ:
                continue
        spo_list.append(tuple(sp))
    d['spo_list'] = spo_list


def seq_padding(X, padding=0):
    L = [len(x) for x in X]
    ML = max(L)
    return np.array([
        np.concatenate([x, [padding] * (ML - len(x))]) if len(x) < ML else x for x in X
    ])


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


def seq_maxpool(x):
    """seq是[None, seq_len, s_size]的格式，
    mask是[None, seq_len, 1]的格式，先除去mask部分，
    然后再做maxpooling。
    """
    seq, mask = x
    seq -= (1 - mask) * 1e10
    return K.max(seq, 1, keepdims=True)


def seq_and_vec(x):
    """seq是[None, seq_len, s_size]的格式，
    vec是[None, v_size]的格式，将vec重复seq_len次，拼到seq上，
    得到[None, seq_len, s_size+v_size]的向量。
    """
    seq, vec = x
    vec = K.expand_dims(vec, 1)
    vec = K.zeros_like(seq[:, :, :1]) + vec
    return K.concatenate([seq, vec], 2)


def dilated_gated_conv1d(seq, mask, dilation_rate=1):
    """膨胀门卷积（残差式）
    """
    dim = K.int_shape(seq)[-1]
    h = Conv1D(dim*2, 3, padding='same', dilation_rate=dilation_rate)(seq)
    def _gate(x):
        dropout_rate = 0.1
        s, h = x
        g, h = h[:, :, :dim], h[:, :, dim:]
        g = K.in_train_phase(K.dropout(g, dropout_rate), g)
        g = K.sigmoid(g)
        return g * s + (1 - g) * h
    seq = Lambda(_gate)([seq, h])
    seq = Lambda(lambda x: x[0] * x[1])([seq, mask])
    return seq


def position_id(x):
    if isinstance(x, list) and len(x) == 2:
        x, r = x
    else:
        r = 0
    pid = K.arange(K.shape(x)[1])
    pid = K.expand_dims(pid, 0)
    pid = K.tile(pid, [K.shape(x)[0], 1])
    return K.abs(pid - K.cast(r, 'int32'))


class Attention(Layer):
    """多头注意力机制
    """
    def __init__(self, nb_head, size_per_head, **kwargs):
        self.nb_head = nb_head
        self.size_per_head = size_per_head
        self.out_dim = nb_head * size_per_head
        super(Attention, self).__init__(**kwargs)
    def build(self, input_shape):
        q_in_dim = input_shape[0][-1]
        k_in_dim = input_shape[1][-1]
        v_in_dim = input_shape[2][-1]
        self.q_kernel = self.add_weight(name='q_kernel',
                                        shape=(q_in_dim, self.out_dim),
                                        initializer='glorot_normal')
        self.k_kernel = self.add_weight(name='k_kernel',
                                        shape=(k_in_dim, self.out_dim),
                                        initializer='glorot_normal')
        self.v_kernel = self.add_weight(name='w_kernel',
                                        shape=(v_in_dim, self.out_dim),
                                        initializer='glorot_normal')
    def mask(self, x, mask, mode='mul'):
        if mask is None:
            return x
        else:
            for _ in range(K.ndim(x) - K.ndim(mask)):
                mask = K.expand_dims(mask, K.ndim(mask))
            if mode == 'mul':
                return x * mask
            else:
                return x - (1 - mask) * 1e10
    def call(self, inputs):
        q, k, v = inputs[:3]
        v_mask, q_mask = None, None
        if len(inputs) > 3:
            v_mask = inputs[3]
            if len(inputs) > 4:
                q_mask = inputs[4]
        # 线性变化
        qw = K.dot(q, self.q_kernel)
        kw = K.dot(k, self.k_kernel)
        vw = K.dot(v, self.v_kernel)
        # 形状变换
        qw = K.reshape(qw, (-1, K.shape(qw)[1], self.nb_head, self.size_per_head))
        kw = K.reshape(kw, (-1, K.shape(kw)[1], self.nb_head, self.size_per_head))
        vw = K.reshape(vw, (-1, K.shape(vw)[1], self.nb_head, self.size_per_head))
        # 维度置换
        qw = K.permute_dimensions(qw, (0, 2, 1, 3))
        kw = K.permute_dimensions(kw, (0, 2, 1, 3))
        vw = K.permute_dimensions(vw, (0, 2, 1, 3))
        # Attention
        a = K.batch_dot(qw, kw, [3, 3]) / self.size_per_head**0.5
        a = K.permute_dimensions(a, (0, 3, 2, 1))
        a = self.mask(a, v_mask, 'add')
        a = K.permute_dimensions(a, (0, 3, 2, 1))
        a = K.softmax(a)
        # 完成输出
        o = K.batch_dot(a, vw, [3, 2])
        o = K.permute_dimensions(o, (0, 2, 1, 3))
        o = K.reshape(o, (-1, K.shape(o)[1], self.out_dim))
        o = self.mask(o, q_mask, 'mul')
        return o
    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], input_shape[0][1], self.out_dim)


class ExponentialMovingAverage:
    """对模型权重进行指数滑动平均。
    用法：在model.compile之后、第一次训练之前使用；
    先初始化对象，然后执行inject方法。
    """
    def __init__(self, model, momentum=0.9999):
        self.momentum = momentum
        self.model = model
        self.ema_weights = [K.zeros(K.shape(w)) for w in model.weights]
    def inject(self):
        """添加更新算子到model.metrics_updates。
        """
        self.initialize()
        for w1, w2 in zip(self.ema_weights, self.model.weights):
            op = K.moving_average_update(w1, w2, self.momentum)
            self.model.metrics_updates.append(op)
    def initialize(self):
        """ema_weights初始化跟原模型初始化一致。
        """
        self.old_weights = K.batch_get_value(self.model.weights)
        K.batch_set_value(zip(self.ema_weights, self.old_weights))
    def apply_ema_weights(self):
        """备份原模型权重，然后将平均权重应用到模型上去。
        """
        self.old_weights = K.batch_get_value(self.model.weights)
        ema_weights = K.batch_get_value(self.ema_weights)
        K.batch_set_value(zip(self.model.weights, ema_weights))
    def reset_old_weights(self):
        """恢复模型到旧权重。
        """
        K.batch_set_value(zip(self.model.weights, self.old_weights))


def anwsers(text_in,question_in):
    _t = [char2id.get(c, 1) for c in text_in]
    _q = [char2id.get(c, 1) for c in question_in]
    _t = np.array([_t])
    _q = np.array([_q])
    _a1, _a2,_an = test_model.predict([_t,_q])
    _an = _an[0]
    _a1, _a2 = _a1[0, :, 0], _a2[0, :, 0]
    return _a1,_a2,_an


class Evaluate(Callback):

    def __init__(self,patience=5):
        super(Evaluate,self).__init__()
        self.F1 = []
        self.best = 0.
        self.wait = 0
        self.patience = patience
    def on_epoch_end(self, epoch, logs=None):
        EMAer.apply_ema_weights()
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
        EMAer.reset_old_weights()

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


data = json.load(open('small_train_data.json',encoding='utf-8')).get('data')
id2char, char2id = json.load(open('all_chars.json',encoding='utf-8'))
train_data,dev_data = train_test_split(np.arange(len(data)),test_size=1)
train_data = list(np.asarray(data)[train_data])
dev_data = list(np.asarray(data)[dev_data])


# input
t_in = Input(shape=(None,))
q_in = Input(shape=(None,))
a1_in = Input(shape=(None,))
a2_in = Input(shape=(None,))
an_in = Input(shape=(4,))

t, q, a1, a2,a = t_in, q_in, a1_in, a2_in, an_in

#question encoder
mask_q = Lambda(lambda x: K.cast(K.greater(K.expand_dims(x, 2), 0), 'float32'))(q)
pid_q = Lambda(position_id)(q)
position_embedding = Embedding(maxlen_q, char_size, embeddings_initializer='zeros')
pv_q = position_embedding(pid_q)
q = Embedding(len(char2id)+2, char_size,name='question_enbeding')(q) # 0: padding, 1: unk
q = Add()([q,pv_q])
q = Dropout(0.25)(q)
q = Lambda(lambda x: x[0] * x[1])([q, mask_q])
q = dilated_gated_conv1d(q, mask_q, 1)
q = dilated_gated_conv1d(q, mask_q, 2)
q = dilated_gated_conv1d(q, mask_q, 5)

h_q = Attention(8, 16)([q, q, q, mask_q])
q_dim = K.int_shape(q)[-1]
k = Bidirectional(GRU(q_dim),merge_mode='mul')(h_q,mask=mask_q)

#text encoder
mask_t = Lambda(lambda x: K.cast(K.greater(K.expand_dims(x, 2), 0), 'float32'))(t)
pid = Lambda(position_id)(t)
position_embedding = Embedding(maxlen_t, char_size, embeddings_initializer='zeros')
pv_t = position_embedding(pid)
t1 = Embedding(len(char2id)+2, char_size)(t) # 0: padding, 1: unk
# t2 = Dense(char_size, use_bias=False)(t2) # 词向量也转为同样维度
t_dim1 = K.int_shape(t1)[-1]
# t = seq_and_vec([t1,k])
t = Lambda(seq_and_vec, output_shape=(None, t_dim1*2))([t1, k])
t = Concatenate()([t,pv_t]) # 字向量、词向量、位置向量相加
t = Dropout(0.25)(t)
t = Lambda(lambda x: x[0] * x[1])([t, mask_t])
t = dilated_gated_conv1d(t, mask_t, 1)
t = dilated_gated_conv1d(t, mask_t, 2)
t = dilated_gated_conv1d(t, mask_t, 5)
t = dilated_gated_conv1d(t, mask_t, 1)
t = dilated_gated_conv1d(t, mask_t, 2)
t = dilated_gated_conv1d(t, mask_t, 5)
t = dilated_gated_conv1d(t, mask_t, 1)
t = dilated_gated_conv1d(t, mask_t, 2)
t = dilated_gated_conv1d(t, mask_t, 5)
t = dilated_gated_conv1d(t, mask_t, 1)
t = dilated_gated_conv1d(t, mask_t, 1)
t = dilated_gated_conv1d(t, mask_t, 1)
t_dim = K.int_shape(t)[-1]

pn1 = Dense(char_size, activation='relu')(t)
pn1 = Dense(1, activation='sigmoid')(pn1)
pn2 = Dense(char_size, activation='relu')(t)
pn2 = Dense(1, activation='sigmoid')(pn2)
an = Conv1D(4,3,activation='sigmoid',name='obj_decoder_fc_po')(t)
an = GlobalAveragePooling1D()(an)

h = Attention(8, 16)([t, t, t, mask_t])
# h = Add()([h,k])
h = Concatenate()([t, h])
#h = Add()([h,k])
h = Conv1D(char_size, 3, activation='relu', padding='same')(h)
ps1 = Dense(1, activation='sigmoid')(h)
ps2 = Dense(1, activation='sigmoid')(h)
an1 = Lambda(lambda x: x[0] * x[1])([ps1, pn1])
an2 = Lambda(lambda x: x[0] * x[1])([ps2, pn2])

#model
train_model = Model([t_in, q_in, a1_in, a2_in, an_in],[an1,an2,an])
test_model = Model([t_in,q_in],[an1,an2,an])
# test_model = Model([t_in,q_in],[an1,an2,an])

# loss
a1 = K.expand_dims(a1, 2)
a2 = K.expand_dims(a2, 2)

s1_loss = K.binary_crossentropy(a1, an1)
s1_loss = K.sum(s1_loss * mask_t) / K.sum(mask_t)
s2_loss = K.binary_crossentropy(a2, an2)
s2_loss = K.sum(s2_loss * mask_t) / K.sum(mask_t)
an_loss = K.categorical_crossentropy(a,an)

loss = an_loss + (s1_loss + s2_loss)

train_model.add_loss(loss)

train_model.compile(optimizer=Adam(1e-3))
train_model.summary(print_fn=logger.info)


EMAer = ExponentialMovingAverage(train_model)
EMAer.inject()


train_D = data_generator(train_data,batch_size=2)
evaluator = Evaluate(patience=50)


if __name__ == '__main__':
    test_model.load_weights('best_model.weights',skip_mismatch=True,by_name=True,reshape=True)
    train_model.fit_generator(train_D.__iter__(),
                              steps_per_epoch=len(train_D),
                              epochs=120,
                              callbacks=[evaluator]
                              )
else:
    train_model.load_weights('best_model.weights')
