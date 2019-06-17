#!/usr/bin/env python
# encoding: utf-8
'''
@author: tianxiaomo
@license: (C) Apache.
@contact: huguanghao520@gmail.com
@software: PyCharm
@file: main.py
@time: 2019/6/15 15:50
@desc:
'''
import json
import torch
from tqdm import tqdm
import numpy as np

import torch.nn as nn
import torch.nn.functional as F


resume = 'saved/ReadCom/resume.tar'
data_file = '../small_train_data.json'
char_path = '../all_chars.json'
result_path = 'all_chars.json'


def seq_and_vec(x,dim):
    """seq是[None, seq_len, s_size]的格式，
    vec是[None, v_size]的格式，将vec重复seq_len次，拼到seq上，
    得到[None, seq_len, s_size+v_size]的向量。
    """
    seq, vec = x
    vec = torch.unsqueeze(vec,dim)
    vec = torch.zeros_like(seq[:, :, :1]) + vec
    return torch.cat([seq, vec], 2)


class RC_demo(nn.Module):

    def __init__(self,maxlen_q,maxlen_t,char_size,char2id):
        super(RC_demo,self).__init__()
        self.question_embedding = nn.Embedding(len(char2id) + 2, char_size)
        self.text_emb = nn.Embedding(len(char2id) + 2, char_size)
        self.dropout = nn.Dropout(0.25)

        self.blstm1 = nn.LSTM(128, 64, bidirectional=True)
        self.blstm2 = nn.LSTM(128, 64, bidirectional=True)
        self.blstm3 = nn.LSTM(128, 64, bidirectional=True)
        self.blstm4 = nn.LSTM(128, 64, bidirectional=True)

        self.dcnn1 = nn.Conv1d(128*3, 128, kernel_size=3, padding=1)
        self.dcnn2 = nn.Conv1d(128, 128, kernel_size=3, padding=2, dilation=2)
        self.dcnn4 = nn.Conv1d(128, 128, kernel_size=3, padding=4, dilation=4)

        self.cnn1 = nn.Conv1d(128, 4, kernel_size=3)
        self.gavgpool = nn.AdaptiveMaxPool1d(1)

        self.fc1 = nn.Linear(128, 1)
        self.fc2 = nn.Linear(128, 1)

    def forward(self,input):
        # question encoder
        t, q = input
        q.long()
        mask_q = torch.gt(q, 0).float()
        q = self.question_embedding(q.long())
        q = self.dropout(q)
        q = torch.mul(q, mask_q.unsqueeze(-1))

        q = q.permute(1,0,2)

        q,_ = self.blstm1(q)
        q,_ = self.blstm2(q)

        q = torch.mul(q, mask_q.permute(1,0).unsqueeze(-1))
        q_max,_ = torch.max(q,dim=0)

        # text encoder
        mask_t = torch.gt(t, 0).float()
        t = self.text_emb(t.long())
        t = self.dropout(t)
        t = torch.mul(t, mask_t.unsqueeze(-1))

        t = t.permute(1,0,2)
        t,_ = self.blstm3(t)
        t,_ = self.blstm4(t)

        t = torch.mul(t, mask_t.permute(1,0).unsqueeze(-1))
        t_max,_ = torch.max(t,dim=0)

        t = t.permute(1,0,2)
        # attention
        h = seq_and_vec([t,t_max], 1)
        h = seq_and_vec([h,q_max], 1)


        h = h.permute(0,2,1)
        # decoder
        h = F.relu(self.dcnn1(h))
        h = F.relu(self.dcnn2(h))
        h = F.relu(self.dcnn4(h))
        an = self.cnn1(h)
        an = self.gavgpool(an).squeeze(-1)

        h = h.permute(0,2,1)
        an1 = F.sigmoid(self.fc1(h))
        an2 = F.sigmoid(self.fc2(h))

        return an1,an2,an


def seq_padding(X, padding=0):
    L = [len(x) for x in X]
    ML = max(L)
    return np.array([
        np.concatenate([x, [padding] * (ML - len(x))]) if len(x) < ML else x for x in X
    ])


def collate_rc(batch):
    t,q = batch
    T, Q = [], []
    for _t,_q in zip(t,q):
        T.append(_t)
        Q.append(_q)

    T = np.array(seq_padding(T))
    Q = np.array(seq_padding(Q))

    T, Q = torch.from_numpy(T), torch.from_numpy(Q)
    return T, Q


def get_data(d):
    d = d['paragraphs'][0]
    text = d['context']
    sp = d['qas']
    t = []
    q = []
    id = []
    for _sp in sp:
        _q = _sp['question']
        _id = _sp['id']
        # if len(sp['answers']) == 0:  # 不回答
        #     a1, a2 = [0] * len(text), [0] * len(text)
        #     an = 1
        # elif sp['answers'][0]['answer_start'] != -1:  # 回答
        #     an1 = sp['answers'][0]['answer_start']
        #     an2 = an1 + len(sp['answers'][0]['text'])
        #
        #     a1, a2 = [0] * len(text), [0] * len(text)
        #     a1[an1] = 1
        #     a2[an2 - 1] = 1
        #     an = 0
        # elif sp['answers'][0]['text'] == 'YES':  # yes
        #     a1, a2 = [0] * len(text), [0] * len(text)
        #     an = 2
        # else:  # yes
        #     a1, a2 = [0] * len(text), [0] * len(text)
        #     an = 3

        T = [char2id.get(c, 1) for c in text]  # 1是unk，0是padding
        Q = [char2id.get(c, 1) for c in _q]

        t.append(T)
        q.append(Q)
        id.append(_id)
    t,q = collate_rc([t,q])
    return t,q,id,text


def main():

    data = json.load(open(data_file, encoding='utf-8')).get('data')
    model.eval()
    result = []
    answers = ['','','YES','NO']

    a = 0

    for d in tqdm(data):

        a += 1
        if a ==15:break

        t,q,id,text = get_data(d)
        a1,a2,an = model([t,q])
        a1 = a1[:,:,0].argmax(-1)
        a2 = a2[:, :, 0].argmax(-1)
        an = an.argmax(-1)
        for _id,_a1,_a2,_an in zip(id,a1,a2,an):
            _resule = {}
            try:
                if _an == 0:
                    e = min(int(_a2+1),len(text))
                    _resule['answer'] = text[_a1:e]
                    _resule['id'] = _id
                else:
                    _resule['answer'] = answers[_an]
                    _resule['id'] = _id
            except BaseException:
                _resule['answer'] = ''
                _resule['id'] = _id

            result.append(_resule)

    f = open(result_path, 'w', encoding='utf-8')
    json.dump(result, f, indent=4, ensure_ascii=False)
    f.close()


if __name__ == '__main__':

    id2char, char2id = json.load(open(char_path, encoding='utf-8'))
    model = RC_demo(maxlen_q=128, maxlen_t=1024, char_size=128, char2id=char2id)

    checkpoint = torch.load(resume, map_location='cpu')

    try:
        model.load_state_dict(checkpoint['state_dict'], strict=True)
    except BaseException:
        for i in list(checkpoint['state_dict'].keys()):
            checkpoint['state_dict'][i[7:]] = checkpoint['state_dict'].pop(i)
        model.load_state_dict(checkpoint['state_dict'], strict=True)

    main()