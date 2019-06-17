#!/usr/bin/env python
# encoding: utf-8
'''
@author: tianxiaomo
@license: (C) Apache.
@contact: huguanghao520@gmail.com
@software: PyCharm
@file: demo_torch.py
@time: 2019/6/10 17:18
@desc:
'''
import torch
from torch.utils.data import Dataset
import torch.utils.data as torchdata
import torch.nn as nn
import torch.nn.functional as F

import random,os,json,argparse
import numpy as np

from base_data_loader import BaseDataLoader
from trainer import Trainer
import log


def seq_padding(X, padding=0):
    L = [len(x) for x in X]
    ML = max(L)
    return np.array([
        np.concatenate([x, [padding] * (ML - len(x))]) if len(x) < ML else x for x in X
    ])


def seq_gather(x):
    """seq是[None, seq_len, s_size]的格式，
    idxs是[None, 1]的格式，在seq的第i个序列中选出第idxs[i]个向量，
    最终输出[None, s_size]的向量。
    """
    seq, idxs = x
    idxs = idxs.type(torch.IntType)
    batch_idxs = torch.arange(0,seq.shape[0])
    batch_idxs = torch.unsqueeze(batch_idxs, 1)
    idxs = torch.cat([batch_idxs,idxs],1)
    return torch.gather(seq,idxs)


def seq_maxpool(x):
    """seq是[None, seq_len, s_size]的格式，
    mask是[None, seq_len, 1]的格式，先除去mask部分，
    然后再做maxpooling。
    """
    seq, mask = x
    seq -= (1 - mask) * 1e10
    return torch.max(seq, 1, keepdims=True)


def seq_and_vec(x,dim):
    """seq是[None, seq_len, s_size]的格式，
    vec是[None, v_size]的格式，将vec重复seq_len次，拼到seq上，
    得到[None, seq_len, s_size+v_size]的向量。
    """
    seq, vec = x
    vec = torch.unsqueeze(vec,dim)
    vec = torch.zeros_like(seq[:, :, :1]) + vec
    return torch.cat([seq, vec], 2)


def position_id(x):
    if isinstance(x, list) and len(x) == 2:
        x, r = x
    else:
        r = 0
    pid = torch.arange(x.shape[1])
    pid = torch.unsqueeze(pid, 0)
    pid = pid.view(1,-1)
    return torch.abs(pid - torch.LongTensor([r]).unsqueeze(0))


class DilatedGatedConv1d(nn.Module):
    '''
    膨胀门卷积（残差式）
    '''
    def __init__(self,inputDim,dilation_rate=1):
        super(DilatedGatedConv1d,self).__init__()
        self.inputDim = inputDim
        self.dcnn = nn.Conv1d(inputDim, inputDim * 2, 3, padding=dilation_rate, dilation=dilation_rate)

    def _gate(self,x,dim):
        dropout_rate = 0.1
        s, h = x
        g, h = h[:,:dim,:],h[:,dim:,:]
        g = g
        g = torch.sigmoid(g)
        return torch.mul(g,s) + torch.mul((1-g) , h)

    def forward(self, seq,mask):
        h = self.dcnn(seq)
        seq = self._gate([seq, h],dim=self.inputDim)
        seq = torch.mul(seq,mask.unsqueeze(1))
        return seq


class Attention(torch.nn.Module):
    """多头注意力机制
    """
    def __init__(self, nb_head, size_per_head,input_shape, **kwargs):
        super(Attention,self).__init__()
        self.nb_head = nb_head
        self.size_per_head = size_per_head
        self.out_dim = nb_head * size_per_head

        q_in_dim = input_shape[0]
        k_in_dim = input_shape[1]
        v_in_dim = input_shape[2]

        self.q_kernel = nn.Parameter(torch.Tensor(self.out_dim,q_in_dim))
        self.k_kernel = nn.Parameter(torch.Tensor(self.out_dim,k_in_dim))
        self.v_kernel = nn.Parameter(torch.Tensor(self.out_dim,v_in_dim))
        nn.init.xavier_uniform_(self.q_kernel)
        nn.init.xavier_uniform_(self.k_kernel)
        nn.init.xavier_uniform_(self.v_kernel)


    def mask(self, x, mask, mode='mul'):
        if mask is None:
            return x
        else:
            for _ in range(len(x.shape) - len(mask.shape)):
                mask = torch.unsqueeze(mask, len(mask.shape))
            if mode == 'mul':
                return x * mask
            else:
                return x - (1 - mask) * 1e10

    def forward (self, inputs):
        q, k, v = inputs[:3]
        v_mask, q_mask = None, None
        if len(inputs) > 3:
            v_mask = inputs[3]
            if len(inputs) > 4:
                q_mask = inputs[4]
        # 线性变化
        qw = torch.matmul(self.q_kernel,q)
        kw = torch.matmul(self.k_kernel,k)
        vw = torch.matmul(self.v_kernel,v)
        # 形状变换
        # qw = torch.reshape(qw, (-1, self.nb_head, self.size_per_head,qw.shape[-1]))
        # kw = torch.reshape(kw, (-1, self.nb_head, self.size_per_head,kw.shape[-1]))
        # vw = torch.reshape(vw, (-1, self.nb_head, self.size_per_head,vw.shape[-1]))
        # # 维度置换
        # qw = qw.permute(0, 3, 1, 2)
        # kw = kw.permute(0, 3, 1, 2)
        # vw = vw.permute(0, 3, 1, 2)
        # Attention
        a = torch.matmul(qw.permute(0, 2, 1), kw) / np.sqrt(128)
        # a = torch.mul(qw, kw) / self.size_per_head**0.5
        # a = a.permute(0, 2, 1, 3)
        a = self.mask(a, v_mask, 'add')
        # a = a.permute(0, 2, 1, 3)
        a = F.softmax(a,dim=-1)
        # 完成输出
        o = torch.matmul(a, vw.permute(0,2,1))
        o = o.permute(0,2,1)
        # o = o.reshape(-1,self.out_dim,o.shape[-1])
        o = self.mask(o, q_mask, 'mul')
        return o


class BidirectionalLSTM(nn.Module):

    def __init__(self, nIn, nHidden, nOut):
        super(BidirectionalLSTM, self).__init__()

        self.rnn = nn.LSTM(nIn, nHidden, bidirectional=True)
        self.embedding = nn.Linear(nHidden * 2, nHidden)

    def forward(self, input,mask=None):
        recurrent, _ = self.rnn(input)
        T, b, h = recurrent.size()
        t_rec = recurrent.view(T * b, h)

        output = self.embedding(t_rec)  # [T * b, nOut]
        output = output.view(b, T, -1).permute([0,2,1])
        output = F.adaptive_avg_pool1d(output,1).squeeze(-1)
        return output


class ReadComDataset(Dataset):

    def __init__(self,data,config,char2id):
        super(ReadComDataset,self).__init__()
        self.data = data
        self.batch_size = config['data_loader']['batch_size']
        self.steps = len(self.data) // self.batch_size
        self.char2id = char2id

    def __getitem__(self, index):
        # T,A1,A2,Q,QA,AN = [],[],[],[],[],[] #文本,问题,答案位置1,答案位置2,解决回答,是,否
        d = self.data[index]['paragraphs'][0]
        text = d['context']
        sp = random.choice(d['qas'])
        q = sp['question']
        if len(sp['answers']) == 0:                 # 不回答
            a1, a2 = [0] * len(text), [0] * len(text)
            an = 1
        elif sp['answers'][0]['answer_start'] != -1:    # 回答
            an1 = sp['answers'][0]['answer_start']
            an2 = an1 + len(sp['answers'][0]['text'])

            a1, a2 = [0] * len(text), [0] * len(text)
            a1[an1] = 1
            a2[an2 - 1] = 1
            an = 0
        elif sp['answers'][0]['text'] == 'YES':         # yes
            a1, a2 = [0] * len(text), [0] * len(text)
            an = 2
        else:                                           # yes
            a1, a2 = [0] * len(text), [0] * len(text)
            an = 3

        T = [self.char2id.get(c, 1) for c in text]  # 1是unk，0是padding
        Q = [self.char2id.get(c, 1) for c in q]
        A1 = a1
        A2 = a2
        AN = an
        return T,Q,A1,A2,AN

    def __len__(self):
        return len(self.data)


class RCLoaderFactory(BaseDataLoader):

    def __init__(self, ds,config):
        super(RCLoaderFactory, self).__init__(config)
        self.workers = config['data_loader']['workers']
        self.split = config['validation']['validation_split']
        self.__trainDataset, self.__valDataset = self.__train_val_split(ds)

    def collate_rc(self,batch):
        T, Q, A1, A2, AN = [], [], [], [], []
        for i in batch:
            T.append(i[0])
            Q.append(i[1])
            A1.append(i[2])
            A2.append(i[3])
            AN.append(i[4])

        T = np.array(seq_padding(T))
        Q = np.array(seq_padding(Q))
        A1 = np.array(seq_padding(A1))
        A2 = np.array(seq_padding(A2))
        AN = np.array(AN)

        T, Q, A1, A2, AN = torch.from_numpy(T), torch.from_numpy(Q), torch.from_numpy(A1), torch.from_numpy(
            A2), torch.from_numpy(AN)
        A1 = A1.unsqueeze(-1)
        A2 = A2.unsqueeze(-1)
        return T, Q, A1, A2, AN

    def train(self):
        trainLoader = torchdata.DataLoader(self.__trainDataset, num_workers = self.num_workers, batch_size = self.batch_size,
                                           shuffle = self.shuffle,collate_fn=self.collate_rc)
        return trainLoader

    def val(self):
        shuffle = False
        valLoader = torchdata.DataLoader(self.__valDataset, num_workers = self.num_workers, batch_size = self.batch_size,
                                         shuffle = shuffle,collate_fn=self.collate_rc)
        return valLoader

    def __train_val_split(self, ds):
        '''
        :param ds: dataset
        :return:
        '''
        split = self.split

        try:
            split = float(split)
        except:
            raise RuntimeError('Train and val splitting ratio is invalid.')

        val_len = int(split * len(ds))
        train_len = len(ds) - val_len
        train, val = torchdata.random_split(ds, [train_len, val_len])
        return train, val

    def split_validation(self):
        raise NotImplementedError


class RCLoss(nn.Module):

    def __init__(self):
        super(RCLoss, self).__init__()
        self.a1loss = nn.BCELoss()
        self.a2loss = nn.BCELoss()
        self.anloss = nn.CrossEntropyLoss()

    def forward(self,a1_p,a2_p,an_p,a1,a2,an):
        a1_loss = self.a1loss(a1_p,a1) * 50
        a2_loss = self.a2loss(a2_p,a2) * 50
        an_loss = self.anloss(an_p,an)
        return a1_loss,a2_loss,an_loss


def RCMetrics(**input):
    a1, a2, an, a1_p, a2_p, an_p = input.get('a1'), input.get('a2'), input.get('an'), input.get('a1_p'), input.get('a2_p'), input.get('an_p')
    _number = np.zeros(4)
    _metrics = np.zeros(4)
    total_metrics = np.zeros(5)

    an_p = np.asarray(an_p.cpu().argmax(-1))
    an = np.asarray(an.cpu())
    a1 = np.asarray(a1[:, :, 0].argmax(-1).cpu())
    a2 = np.asarray(a2[:, :, 0].argmax(-1).cpu())
    a1_p = np.asarray(a1_p[:, :, 0].argmax(-1).cpu())
    a2_p = np.asarray(a2_p[:, :, 0].argmax(-1).cpu())

    for i in range(1,4):
        _metrics[i] = len([i for i, j in zip(an_p.__eq__(i), an.__eq__(i)) if
             i == True and j == True])

    for i in range(4):
        _number[i] = an.__eq__(i).sum()

    for t1, t2, ta, p1, p2, pa in zip(a1, a2, an, a1_p, a2_p, an_p):
        if ta == 0 and pa == 0 and p1 == t1 and t2 == p2:  # 回答
            _metrics[0] += 1

    for i in range(4):
        if _number[i] != 0:
            total_metrics[i] = _metrics[i] / _number[i]
    total_metrics[4] = _metrics.sum() / _number.sum()

    return total_metrics


class RC(nn.Module):

    def __init__(self,maxlen_q,maxlen_t,char_size,char2id):
        super(RC,self).__init__()
        self.position_embedding = nn.Embedding(maxlen_q, char_size)
        self.question_embedding = nn.Embedding(len(char2id) + 2, char_size)
        self.dropout = nn.Dropout(0.25)
        self.dgcnn1 = DilatedGatedConv1d(128)

        self.attention_q = Attention(8,16,input_shape=[128,128,128])
        self.blstm = BidirectionalLSTM(128,128,128)
        self.cnn1 = nn.Conv1d(128,4,kernel_size=3)
        self.gavgpool = nn.AdaptiveMaxPool1d(1)

        self.text_position_emb = nn.Embedding(maxlen_t, char_size)
        self.text_emb = nn.Embedding(len(char2id) + 2, char_size)
        t_m = 3
        self.dgcnn_t_1_1 = DilatedGatedConv1d(128*t_m, dilation_rate=1)
        self.dgcnn_t_1_2 = DilatedGatedConv1d(128*t_m, dilation_rate=2)
        self.dgcnn_t_1_5 = DilatedGatedConv1d(128*t_m, dilation_rate=5)
        self.dgcnn_t_2_1 = DilatedGatedConv1d(128*t_m, dilation_rate=1)
        self.dgcnn_t_2_2 = DilatedGatedConv1d(128*t_m, dilation_rate=2)
        self.dgcnn_t_2_5 = DilatedGatedConv1d(128*t_m, dilation_rate=5)
        self.dgcnn_t_3_1 = DilatedGatedConv1d(128*t_m, dilation_rate=1)
        self.dgcnn_t_3_2 = DilatedGatedConv1d(128*t_m, dilation_rate=2)
        self.dgcnn_t_3_5 = DilatedGatedConv1d(128*t_m, dilation_rate=5)
        self.dgcnn_t_4_10 = DilatedGatedConv1d(128*t_m, dilation_rate=1)
        self.dgcnn_t_4_11 = DilatedGatedConv1d(128*t_m, dilation_rate=1)
        self.dgcnn_t_4_12 = DilatedGatedConv1d(128*t_m, dilation_rate=1)

        self.fc_a1 = nn.Linear(128*t_m, 128)
        self.fc1_a1 = nn.Linear(128, 1)
        self.fc_a2 = nn.Linear(128*t_m, 128)
        self.fc1_a2 = nn.Linear(128, 1)

        self.cnn1 = nn.Conv1d(128*t_m, 4, kernel_size=3)
        self.gavgpool = nn.AdaptiveMaxPool1d(1)

        self.attention_t = Attention(8, 16, input_shape=[128*t_m, 128*t_m, 128*t_m])
        self.cnn2 = nn.Conv1d(128*(t_m + 1), 128, kernel_size=3,padding=1)
        self.fc1 = nn.Linear(128, 1)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, input):
        # question encoder
        t,q = input
        q.long()
        mask_q = torch.gt(q, 0).float()
        pid_q = position_id(q).cuda()
        pv_q = self.position_embedding(pid_q)
        q = self.question_embedding(q.long())
        q = torch.add(q, pv_q)
        q = self.dropout(q)
        q = torch.mul(q,mask_q.unsqueeze(-1))

        q = q.permute([0,2,1])

        q = self.dgcnn1(q, mask_q)
        q = self.dgcnn1(q, mask_q)
        q = self.dgcnn1(q, mask_q)

        h_q = self.attention_q([q, q, q, mask_q])

        h_q = h_q.permute([2,0,1])
        k = self.blstm(h_q, mask=mask_q)

        # text encoder
        mask_t = torch.gt(t, 0).float()
        pid_t = position_id(t).cuda()
        pv_t = self.text_position_emb(pid_t)
        t = self.text_emb(t.long())

        t = seq_and_vec([t, k],1)
        t =  seq_and_vec([t, pv_t.squeeze(0)],0)  # 字向量、词向量、位置向量相加

        t = self.dropout(t)
        t = torch.mul(t, mask_t.unsqueeze(-1))

        t = t.permute([0,2,1])

        t = self.dgcnn_t_1_1(t, mask_t)
        t = self.dgcnn_t_1_2(t, mask_t)
        t = self.dgcnn_t_1_5(t, mask_t)

        an = self.cnn1(t)
        an = self.gavgpool(an)
        an = an.squeeze(-1)

        t = self.dgcnn_t_2_1(t, mask_t)
        t = self.dgcnn_t_2_2(t, mask_t)
        t = self.dgcnn_t_2_5(t, mask_t)
        t = self.dgcnn_t_3_1(t, mask_t)
        t = self.dgcnn_t_3_2(t, mask_t)
        t = self.dgcnn_t_3_5(t, mask_t)
        t = self.dgcnn_t_4_10(t, mask_t)
        t = self.dgcnn_t_4_11(t, mask_t)
        t = self.dgcnn_t_4_12(t, mask_t)
        t_dim = t.shape[1]

        t_p = t.permute([0,2,1])
        pn1 = F.relu(self.fc_a1(t_p))
        pn1 = F.sigmoid(self.fc1_a1(pn1))

        pn2 = F.relu(self.fc_a2(t_p))
        pn2 = F.sigmoid(self.fc1_a2(pn2))

        h = self.attention_t([t, t, t, mask_t])
        h = torch.cat([t, h],dim=1)

        h = self.cnn2(h)
        h_p = h.permute([0,2,1])
        ps1 = F.sigmoid(self.fc1(h_p))
        ps2 = F.sigmoid(self.fc2(h_p))
        an1 = torch.mul(ps1, pn1)
        an2 = torch.mul(ps2, pn2)

        return an1,an2,an


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

        self.dcnn11 = nn.Conv1d(128, 128, kernel_size=3, padding=1)
        self.dcnn12 = nn.Conv1d(128, 128, kernel_size=3, padding=2, dilation=2)
        self.dcnn14 = nn.Conv1d(128, 128, kernel_size=3, padding=4, dilation=4)

        self.cnn1 = nn.Conv1d(128, 4, kernel_size=3)
        self.gavgpool = nn.AdaptiveMaxPool1d(1)

        self.attention = Attention(8,16,input_shape=[128,128,128])

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
        # q_max,_ = torch.max(q,dim=0)

        q_max = q.mean(dim=0)

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
        # h = F.relu(self.dcnn11(h))
        # h = F.relu(self.dcnn12(h))
        # h = F.relu(self.dcnn14(h))
        #
        # q = q.permute(1,2,0)
        # h = self.attention([h, q, q, mask_t])

        an = self.cnn1(h)
        an = self.gavgpool(an).squeeze(-1)

        h = h.permute(0,2,1)
        an1 = F.sigmoid(self.fc1(h))
        an2 = F.sigmoid(self.fc2(h))

        return an1,an2,an


if __name__ == '__main__':
    SEED = 2019
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    torch.backends.cudnn.deterministic = True

    parser = argparse.ArgumentParser(description='PyTorch Template')
    parser.add_argument('-c', '--config', default='config.json', type=str,
                        help='config file path (default: None)')
    parser.add_argument('-r', '--resume', default=None, type=str,
                        help='path to latest checkpoint (default: None)')
    args = parser.parse_args()
    config = json.load(open(args.config))

    logger = log.init_logger(log_path='logs')
    train_logger = logger

    data = json.load(open('../small_train_data.json', encoding='utf-8')).get('data')
    id2char, char2id = json.load(open('../all_chars.json', encoding='utf-8'))

    RCData = ReadComDataset(data,config=config,char2id = char2id)
    data_loader = RCLoaderFactory(RCData,config=config)
    train = data_loader.train()
    val = data_loader.val()

    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(i) for i in config['gpus'])
    # model = eval(config['arch'])(3, 3
    # model = RC(maxlen_q=128,maxlen_t=1024,char_size=128,char2id=char2id)
    model = RC_demo(maxlen_q=128, maxlen_t=1024, char_size=128, char2id=char2id)
    # model.summary()

    loss = RCLoss()
    metrics = RCMetrics
    resume = args.resume

    trainer = Trainer(model, loss, metrics,
                        config=config,
                        resume=resume,
                        data_loader=train,
                        valid_data_loader=val,
                        train_logger=train_logger,
                      )

    trainer.train()
