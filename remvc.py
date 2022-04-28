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
from configparser import ConfigParser

import sys
import os
import time
import json
import pickle

from remvc_tasks import lu_classify, predict_popus
from remvc_flow import FLOW_SSL
from remvc_poi import POI_SSL
from remvc_data import SSLData

FType = torch.FloatTensor
LType = torch.LongTensor

from configparser import ConfigParser
config = ConfigParser()
config.read('conf', encoding='UTF-8')
GPU_DEVICE = config['DEFAULT'].get("GPU_DEVICE")
device = torch.device("cuda:"+GPU_DEVICE if torch.cuda.is_available() else "cpu")
extractor = config['DEFAULT'].get("EXTRACTOR")

size = int(config['DEFAULT'].get("EMB"))
mutual_reg = float(config['DEFAULT'].get("MUTUAL"))
poi_reg = float(config['DEFAULT'].get("REG"))

fw = open("model_result/" + extractor + "_emb_" + str(size), "w")

class Model_SSL():

    def __init__(self): 
        super(Model_SSL,self).__init__()

        self.ssl_data = SSLData()
        self.poi_model = POI_SSL(self.ssl_data, neg_size=10, emb_size=size, attention_size=16, temp=0.08, extractor=extractor).to(device)
        self.flow_model = FLOW_SSL(self.ssl_data, neg_size=150, emb_size=size, temp=0.08, time_zone=48, extractor=extractor).to(device)

        self.epoch = 200
        self.learning_rate = 0.001

        self.mutual_reg = mutual_reg
        self.poi_reg = poi_reg

        self.mutual_neg_size = 5
        self.emb_size = size
        self.init_basic_conf()

        self.opt = Adam(lr=self.learning_rate, params=[{"params":self.poi_model.poi_net.parameters()},\
            {"params":self.flow_model.pickup_net.parameters()}, \
            {"params":self.flow_model.dropoff_net.parameters()}, {"params":self.mutual_net.parameters()}], weight_decay=1e-5)

    def init_basic_conf(self):
        self.mutual_net = torch.nn.Sequential(
                    nn.Linear(self.emb_size*2, 1)).to(device)

    def forward(self, base_poi_emb, base_flow_emb, neg_poi_emb, neg_flow_emb):
        pos_emb = torch.cat([base_poi_emb, base_flow_emb])
        pos_scores = self.mutual_net(pos_emb)
        pos_label = torch.Tensor([1 for _ in range(pos_scores.size(0))]).type(FType).to(device)

        weights = torch.ones(neg_poi_emb.size()[0])
        _indexs = torch.multinomial(weights, self.mutual_neg_size)
        neg_poi_emb = neg_poi_emb[_indexs]
        base_flow_emb = base_flow_emb.repeat(self.mutual_neg_size, 1)
        neg_emb_p = torch.cat([neg_poi_emb, base_flow_emb], dim=1)

        weights = torch.ones(neg_flow_emb.size()[0])
        _indexs = torch.multinomial(weights, self.mutual_neg_size)
        neg_flow_emb = neg_flow_emb[_indexs]
        base_poi_emb = base_poi_emb.repeat(self.mutual_neg_size, 1)
        neg_emb_f = torch.cat([base_poi_emb, neg_flow_emb], dim=1)

        neg_emb = torch.cat([neg_emb_p, neg_emb_f], dim=0)
        neg_scores = self.mutual_net(neg_emb).squeeze()
        neg_label = torch.Tensor([0 for _ in range(neg_scores.size(0))]).type(FType).to(device)

        scores = torch.cat([pos_scores, neg_scores])
        labels = torch.cat([pos_label, neg_label])

        loss = -(F.log_softmax(scores, dim=0) * labels).sum() / labels.sum()

        return loss

    def model_train(self):
        for epoch in range(self.epoch):
            self.loss =  0.0

            for region_id in self.ssl_data.sampling_pool: 
                poi_loss, base_poi_emb, neg_poi_emb = self.poi_model.model_train(region_id)
                flow_loss, base_flow_emb, neg_flow_emb =  self.flow_model.model_train(region_id)
                mutual_loss = self.forward(base_poi_emb, base_flow_emb, neg_poi_emb, neg_flow_emb)

                loss = flow_loss + self.poi_reg * poi_loss + self.mutual_reg * mutual_loss

                self.opt.zero_grad()
                self.loss += loss
                loss.backward()
                self.opt.step()

            print("=============================> iter epoch", epoch)
            print("avg loss = " + str(self.loss))

            if epoch >= 150:
                self.test()
                fw.write("=============================> iter epoch " + str(epoch) + "\n")
                fw.write("avg loss = " + str(self.loss) + "\n")

    def get_emb(self):
        output_flow = self.flow_model.get_emb()
        output_poi = self.poi_model.get_emb()
        output = np.concatenate((output_flow, output_poi), axis=1)
        return output

    def test(self):
        output = self.get_emb()
        lu_classify(output, fw, _type="con")
        predict_popus(output, fw)
        fw.flush()

if __name__ == '__main__':
    model = Model_SSL()
    model.model_train()