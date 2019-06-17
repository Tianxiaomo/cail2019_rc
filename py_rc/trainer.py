#!/usr/bin/env python
# encoding: utf-8
'''
@author: tianxiaomo
@license: (C) Apache.
@contact: huguanghao520@gmail.com
@software: PyCharm
@file: trainer.py
@time: 2019/5/18 14:42
@desc:
'''
import numpy as np
import torch
from base_trainer import BaseTrainer

class Trainer(BaseTrainer):
    """
    Trainer class

    Note:
        Inherited from BaseTrainer.
        self.optimizer is by default handled by BaseTrainer based on config.
    """
    def __init__(self, model, loss, metrics, resume, config,
                 data_loader, valid_data_loader=None, train_logger=None):
        super(Trainer, self).__init__(model, loss, metrics, resume, config, train_logger)
        self.config = config
        self.batch_size = data_loader.batch_size
        self.data_loader = data_loader
        self.valid_data_loader = valid_data_loader
        self.valid = True if self.valid_data_loader is not None else False
        self.log_step = int(np.sqrt(self.batch_size))

    def _to_tensor(self, *tensors):
        t = []
        for __tensors in tensors:
            if __tensors.shape[0] == 0:
                t.append(__tensors)
            else:
                t.append(__tensors.to(self.device))
        return t

    def _eval_metrics(self, output, target, target_lenght):
        acc_metrics = np.zeros(len(self.metrics))
        output = output.cpu().data.numpy()
        target = target.cpu().data.numpy()
        target_l = target_lenght.cpu().data.numpy()
        output = np.argmax(output, axis=2)
        for i, metric in enumerate(self.metrics):
            acc_metrics[i] += metric(output, target,target_l)
        return acc_metrics

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Current training epoch.
        :return: A log that contains all information you want to save.

        Note:
            If you have additional information to record, for example:
                > additional_log = {"x": x, "y": y}
            merge it with log before return. i.e.
                > log = {**log, **additional_log}
                > return log

            The metrics in log must have the key 'metrics'.
        """
        self.model.train()

        total_loss = 0

        total_metrics = np.zeros(5)
        for batch_idx, gt in enumerate(self.data_loader):
            t,q,a1,a2,an = gt
            if torch.cuda.is_available() and self.config['cuda'] and t.shape[0] <= len(self.config['gpus']):
                continue
            # self.logger.debug('img:{0} target:{1} '.format(list(t.shape),list(target.shape)))

            t,q,a1,a2,an = self._to_tensor(t,q,a1,a2,an)

            self.optimizer.zero_grad()
            a1_p,a2_p,an_p = self.model([t,q])

            a1_loss,a2_loss,an_loss = self.loss(a1_p,a2_p,an_p,a1.float(),a2.float(),an.long())
            loss = a1_loss  + a2_loss  + an_loss
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

            # 统计训练信息
            total_metrics += self.metrics(a1=a1, a2=a2, an=an, a1_p=a1_p, a2_p=a2_p, an_p=an_p)

            if self.verbosity >= 2 and batch_idx % self.log_step == 0:
                self.logger.info('Train Epoch: {} [{}/{} ({:.0f}%)] ToTal_Loss: {:.5f} A1_Loss: {:.3f} A2_Loss: {:.3f} AN_Loss: {:.3f} ACC: {:.5f}'.format(
                    epoch,
                    batch_idx * self.data_loader.batch_size,
                    len(self.data_loader) * self.data_loader.batch_size,
                    100.0 * batch_idx / len(self.data_loader),
                    loss.item(),
                    a1_loss.item(),
                    a2_loss.item(),
                    an_loss.item(),
                    total_metrics[4]/(batch_idx+1),
                    ))

        log = {
            'loss': total_loss / len(self.data_loader),
            'metrics': (total_metrics /len(self.data_loader)).tolist()
        }

        if self.valid:
            val_log = self._valid_epoch()
            log = {**log, **val_log}

        return log

    def _valid_epoch(self):
        """
        Validate after training an epoch

        :return: A log that contains information about validation

        Note:
            The validation metrics in log must have the key 'val_metrics'.
        """
        self.model.eval()
        total_val_loss = 0
        total_val_metrics = np.zeros(5)
        with torch.no_grad():
            for batch_idx, gt in enumerate(self.valid_data_loader):
                t, q, a1, a2, an = gt
                if torch.cuda.is_available() and self.config['cuda'] and t.shape[0] <= len(self.config['gpus']):
                    continue

                t, q, a1, a2, an = self._to_tensor(t, q, a1, a2, an)

                self.optimizer.zero_grad()
                a1_p, a2_p, an_p = self.model([t, q])

                a1_loss, a2_loss, an_loss = self.loss(a1_p, a2_p, an_p, a1.float(), a2.float(), an.long())
                loss = a1_loss + a2_loss + an_loss
                total_val_loss += loss.item()

                # 统计训练信息
                total_val_metrics += self.metrics(a1=a1, a2=a2, an=an, a1_p=a1_p, a2_p=a2_p, an_p=an_p)

        return {
            'val_loss': total_val_loss / len(self.valid_data_loader),
            'val_metrics': (total_val_metrics/len(self.valid_data_loader)).tolist()
        }
