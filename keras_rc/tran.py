#!/usr/bin/env python
# encoding: utf-8
'''
@author: tianxiaomo
@license: (C) Apache.
@contact: huguanghao520@gmail.com
@software: PyCharm
@file: tran.py
@time: 2019/6/4 10:50
@desc:
'''
import json

f = open('small_train_data.json', 'r', encoding='utf-8')
data = json.load(f)

se = set()
for i in data.get('data'):
    for j in i.get('paragraphs')[0].get('context'):
        se.add(j)
    for j in i.get('paragraphs')[0].get('casename'):
        se.add(j)
    for q in i.get('paragraphs')[0].get('qas'):
        for j in q.get('question'):
            se.add(j)

char2id = {}
id2char = {}
for i, c in enumerate(se):
    char2id[c] = i + 2
    id2char[i+2] = c

f = open('all_chars.json', 'w', encoding='utf-8')
json.dump([id2char, char2id], f, indent=4, ensure_ascii=False)
f.close()

