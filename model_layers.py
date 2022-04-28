import numpy as np 
import time
import random
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.optim import SGD, Adam, ASGD, RMSprop
from torch.utils.data import DataLoader
from torch.nn.functional import log_softmax, softmax
import torch.nn.functional as F

import math

from configparser import ConfigParser
config = ConfigParser()
config.read('conf', encoding='UTF-8')
_type = config['DEFAULT'].get("TYPE")
func = config['DEFAULT'].get("FUNC")

class SAEncoder(nn.Module):

    def __init__(self, d_input, d_model, n_head):
        super(SAEncoder, self).__init__()

        self.d_input = d_input
        self.d_model = d_model
        self.n_head = n_head

        self.linear_k = nn.Linear(self.d_input, self.d_model * self.n_head) 
        self.linear_v = nn.Linear(self.d_input, self.d_model * self.n_head) 
        self.linear_q = nn.Linear(self.d_input, self.d_model * self.n_head)

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()

    def fusion(self, v, f_type):
        if f_type == "concat":
            output = v.view(-1, self.d_model * self.n_head)
        if f_type == "avg":
            output = torch.mean(v, dim=0)
        return output
    
    def forward(self, x):
        q = self.linear_q(x) 
        k = self.linear_k(x)
        v = self.linear_v(x)

        q_ = q.view(self.n_head, self.d_model) 
        k_ = k.view(self.n_head, self.d_model)
        v_ = v.view(self.n_head, self.d_model)

        head, d_tensor = k_.size()
        score = (q_.matmul(k_.transpose(0, 1))) / math.sqrt(d_tensor)
        score = self.softmax(score)

        v_ = self.relu(v_)
        v = score.matmul(v_)

        output = self.fusion(v, _type)
        return output


class CroSAEncoder(nn.Module):

    def __init__(self, d_input_query, d_input_kv, d_model, n_head):
        super(SAEncoder, self).__init__()

        self.d_input_query = d_input_query
        self.d_input_kv = d_input_kv
        self.d_model = d_model
        self.n_head = n_head

        self.linear_k = nn.Linear(self.d_input_kv, self.d_model * self.n_head) 
        self.linear_v = nn.Linear(self.d_input_kv, self.d_model * self.n_head) 
        self.linear_q = nn.Linear(self.d_input_query, self.d_model * self.n_head)

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()

    def fusion(self, v, f_type):
        if f_type == "concat":
            output = v.view(-1, self.d_model * self.n_head)
        if f_type == "avg":
            output = torch.mean(v, dim=0)
        return output
    
    def forward(self, q, kv):
        q = self.linear_q(q) 
        k = self.linear_k(kv)
        v = self.linear_v(kv)

        q_ = q.view(self.n_head, self.d_model) 
        k_ = k.view(self.n_head, self.d_model)
        v_ = v.view(self.n_head, self.d_model)

        head, d_tensor = k_.size()
        score = (q_.matmul(k_.transpose(0, 1))) / math.sqrt(d_tensor)
        score = self.softmax(score)
       
        v_ = self.relu(v_)
        v = score.matmul(v_)

        output = self.fusion(v, _type)
        if func == "relu":
            output = self.relu(output)

        return output