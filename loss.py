import random
from random import randrange, uniform
import torch.nn as nn
import torch
import numpy as np

class CustomLoss(object):
    def __init__(self, train_on_gpu=True):
        self.train_on_gpu = train_on_gpu

    def real_loss(self, D_out):
        '''Calculates how close discriminator outputs are to being real.
        param, D_out: discriminator logits
        return: real loss'''
        batch_size = D_out.size(0)
        # label smoothing
        labels = torch.ones(batch_size) * np.random.uniform(0.7, 1.2)
        # labels = torch.ones(batch_size) * 0.9
        # labels = torch.ones(batch_size) # real labels = 1
        if self.train_on_gpu:
            labels = labels.cuda()

        criterion = nn.BCEWithLogitsLoss()
        loss = criterion(D_out.squeeze(), labels)
        # loss = torch.mean(-loss)
        # loss = torch.mean((D_out-1)**2)
        return loss

    def fake_loss(self, D_out):
        '''Calculates how close discriminator outputs are to being fake.
        param, D_out: discriminator logits
        return: fake loss'''
        batch_size = D_out.size(0)
        # labels = torch.zeros(batch_size) + 0.0 # fake labels = 0
        labels = torch.zeros(batch_size) * np.random.uniform(0.0, 0.3) # fake labels = 0.3
        # print(labels)
        if self.train_on_gpu:
            labels = labels.cuda()
        criterion = nn.BCEWithLogitsLoss()
        loss = criterion(D_out.squeeze(), labels)
        # loss = torch.mean(loss)
        # loss = torch.mean(D_out**2)
        return loss