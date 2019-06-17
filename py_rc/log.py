#!/usr/bin/env python
# encoding: utf-8
'''
@author: tianxiaomo
@license: (C) Apache.
@contact: huguanghao520@gmail.com
@software:
@file: log.py
@time: 2019/3/17 16:29
@desc:
'''
import datetime
import logging
import os
import sys

def get_date_str():
    now = datetime.datetime.now()
    return now.strftime('%Y-%m-%d_%H-%M-%S')

def init_logger(log_file=None, log_path=None, log_level=logging.DEBUG, mode='w', stdout=True):
    """
    log_path: 日志文件的文件夹路径
    mode: 'a', append; 'w', 覆盖原文件写入.
    """
    fmt = '%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s: %(message)s'
    if log_path is None:
        log_path = '~/temp/log/'
    if log_file is None:
        log_file = 'log_' + get_date_str() + '.log'
    log_file = os.path.join(log_path, log_file)
    # 此处不能使用logging输出
    print('log file path:' + log_file)
    if not os.path.exists(log_path):
        os.mkdir(log_path)
    logging.basicConfig(level=log_level,
                        format=fmt,
                        filename=log_file,
                        filemode=mode)

    if stdout:
        console = logging.StreamHandler(stream=sys.stdout)
        console.setLevel(log_level)
        formatter = logging.Formatter(fmt)
        console.setFormatter(formatter)
        logging.getLogger('').addHandler(console)

    return logging