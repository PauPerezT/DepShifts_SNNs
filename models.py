# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
->models.py

This file was created for the <<Longitudinal Modeling of Depression Shifts Using Speech and Language>> paper in the ICASSP 2024 conference
a collaborative work of KCL, FAU, and the RADAr consortium.

Original file author: PAULA ANDREA PEREZ-TORO
@email:paula.andrea.perez@fau.de


"""
import torch
from torch import nn, optim
import torch.nn.functional as F
# from echotorch.nn.LiESNCell import LiESNCell
import math
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict
from .attentions import *
from torch.autograd import Variable
import torchvision

device = torch.device('cuda' if torch.cuda.is_available() else "cpu")





class resnetGRU_Siamese(nn.Module):
    def __init__(self, input_dim,input_channel=1, num_classes=1000):
        self.inplanes = 64
        super(resnetGRU_Siamese, self).__init__()
        self.gru = nn.GRU(input_size=input_dim, hidden_size=128)
        self.resnet = torchvision.models.resnet18(False)  # Initializing resnet18


        self.resnet.conv1 = nn.Conv2d(input_channel, 64, kernel_size=7, stride=2, padding=3,
                                      bias=False)
        num_ftrs = self.resnet.fc.in_features  # Getting last layer's output features
        self.resnet.fc = nn.Linear(num_ftrs, 512)  # Modifying the last layer accordng to our need
        self.fc = nn.Linear(512 , num_classes)

    def net_def(self, x):
        # x = F.normalize(x, p=2)
        x, _ = self.gru(torch.transpose(F.normalize(x, p=2), 1, 2))  # GRU Layer
        x = self.resnet(torch.transpose(x, 1, 2).unsqueeze(1))

        return x

    def forward(self, input, cnt=True):

        res = []
        res_sig = []
        if cnt:
            for i in range(2):
                x = input[:, i].to(device).type(torch.float)

                # x = F.normalize(x, p=2) #Unit Normalization Layer
                x = F.gelu(self.net_def(x))
                res.append(x)


            #
            x=res[1]-res[0]
            x = self.fc(x)


            return res, x

        else:
            x = self.net_def(input)
            x = self.classlinear2(x)
            x = F.softmax(x)

            return x

