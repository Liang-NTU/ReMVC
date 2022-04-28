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
from sklearn.metrics.pairwise import cosine_similarity

import sys
import os
import time
import json
import pickle

from remvc_tasks import lu_classify, predict_popus
from remvc_data import SSLData
from model_layers import SAEncoder

FType = torch.FloatTensor
LType = torch.LongTensor

from configparser import ConfigParser
config = ConfigParser()
config.read('conf', encoding='UTF-8')
GPU_DEVICE = config['DEFAULT'].get("GPU_DEVICE")
device = torch.device("cuda:"+GPU_DEVICE if torch.cuda.is_available() else "cpu")

rs_type = "random"

class Flatten(torch.nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class FLOW_SSL(torch.nn.Module):

    def __init__(self, ssl_data, neg_size, emb_size, temp, time_zone, extractor): 
        super(FLOW_SSL,self).__init__()

        self.ssl_data = ssl_data
        self.init_basic_conf(neg_size, emb_size, temp, time_zone, extractor)
        
    def init_basic_conf(self, neg_size, emb_size, temp, time_zone, extractor):
        self.neg_size = neg_size
        self.emb_size = emb_size
        self.temp = temp
        self.time_zone = time_zone

        self.extractor = extractor

        if self.extractor == "CNN":
            self.pickup_net = torch.nn.Sequential(
                nn.Conv1d(in_channels=self.time_zone, out_channels=4, kernel_size=7),
                Flatten(),
                nn.Linear(1056, self.emb_size)
            ).to(device)

            self.dropoff_net = torch.nn.Sequential(
                nn.Conv1d(in_channels=self.time_zone, out_channels=4, kernel_size=7),
                Flatten(),
                nn.Linear(1056, self.emb_size)
            ).to(device)

        if self.extractor == "MLP":
            self.pickup_net = torch.nn.Sequential(
                nn.Linear(270, self.emb_size),
                nn.ReLU(),
            ).to(device)

            self.dropoff_net = torch.nn.Sequential(
                nn.Linear(270, self.emb_size),
                nn.ReLU(),
            ).to(device)

        if self.extractor == "SA":
            self.pickup_net = SAEncoder(d_input=270, d_model=16, n_head=3).to(device)
            self.dropoff_net = SAEncoder(d_input=270, d_model=16, n_head=3).to(device)


    def gaussian_noise(self, matrix, mean=0, sigma=0.03):
        matrix = matrix.copy()
        noise = np.random.normal(mean, sigma, matrix.shape)
        mask_overflow_upper = matrix+noise >= 1.0
        mask_overflow_lower = matrix+noise < 0
        noise[mask_overflow_upper] = 1.0
        noise[mask_overflow_lower] = 0
        matrix += noise
        return matrix

    def positive_sampling(self, region_id):
        pos_flow_sets = []
        _, flow_matrix = self.ssl_data.get_region(region_id)
        pickup_matrix, dropoff_matrix = flow_matrix

        for sigma in [0.0001, 0.0001, 0.0001, 0.0001]:
            pickup_matrix = self.gaussian_noise(pickup_matrix, sigma=sigma)
            dropoff_matrix = self.gaussian_noise(dropoff_matrix, sigma=sigma)
            pos_flow_sets.append([pickup_matrix, dropoff_matrix])

        return pos_flow_sets

    def negative_sampling(self, region_id):
        sampling_pool = []
        for _id in self.ssl_data.sampling_pool:
            if _id == region_id:
                continue
            sampling_pool.append(_id)

        p = self.ssl_data.rs_ratio["model_flow"][region_id]
        neg_region_ids = np.random.choice(sampling_pool, self.neg_size, replace=False, p=p)

        neg_flow_sets = []
        for neg_region_id in neg_region_ids:
            _, flow_matrix = self.ssl_data.get_region(neg_region_id)
            neg_flow_sets.append(flow_matrix)


        return neg_flow_sets

    def agg_region_emb(self, flow_matrix):
        pickup_matrix = flow_matrix[0]
        dropoff_matrix = flow_matrix[1]

        if self.extractor == "CNN":
            pickup_matrix = torch.from_numpy(pickup_matrix).type(FType).to(device)
            pickup_matrix = pickup_matrix.unsqueeze(0)
            pickup_emb = self.pickup_net(pickup_matrix)

            dropoff_matrix = torch.from_numpy(dropoff_matrix).type(FType).to(device)
            dropoff_matrix = dropoff_matrix.unsqueeze(0)
            dropoff_emb = self.dropoff_net(dropoff_matrix)

        if self.extractor == "MLP":
            pickup_matrix = np.sum(pickup_matrix, axis=0)
            pickup_matrix = torch.from_numpy(pickup_matrix).type(FType).to(device)
            pickup_emb = self.pickup_net(pickup_matrix)

            dropoff_matrix = np.sum(dropoff_matrix, axis=0)
            dropoff_matrix = torch.from_numpy(dropoff_matrix).type(FType).to(device)
            dropoff_emb = self.dropoff_net(dropoff_matrix)

        if self.extractor == "SA":
            pickup_matrix = np.sum(pickup_matrix, axis=0)
            pickup_matrix = torch.from_numpy(pickup_matrix).type(FType).to(device)
            pickup_emb = self.pickup_net(pickup_matrix)

            dropoff_matrix = np.sum(dropoff_matrix, axis=0)
            dropoff_matrix = torch.from_numpy(dropoff_matrix).type(FType).to(device)
            dropoff_emb = self.dropoff_net(dropoff_matrix)

        # region_emb = torch.cat([pickup_emb, dropoff_emb], dim=1).squeeze()
        region_emb = (pickup_emb + dropoff_emb) / 2
        region_emb = region_emb.squeeze()

        return region_emb

    def forward(self, flow_matrix, pos_flow_sets, neg_flow_sets):
        base_region_emb = self.agg_region_emb(flow_matrix)

        pos_region_emb_list = []
        for pos_flow_matrix in pos_flow_sets:
            pos_region_emb = self.agg_region_emb(pos_flow_matrix)
            pos_region_emb_list.append(pos_region_emb.unsqueeze(0))
        pos_region_emb = torch.cat(pos_region_emb_list, dim=0)

        neg_region_emb_list = []
        for neg_flow_matrix in neg_flow_sets:
            neg_region_emb = self.agg_region_emb(neg_flow_matrix)
            neg_region_emb_list.append(neg_region_emb.unsqueeze(0))
        neg_region_emb = torch.cat(neg_region_emb_list, dim=0)

        pos_scores = torch.matmul(pos_region_emb, base_region_emb)
        pos_label = torch.Tensor([1 for _ in range(pos_scores.size(0))]).type(FType).to(device)
        
        neg_scores = torch.matmul(neg_region_emb, base_region_emb)
        neg_label = torch.Tensor([0 for _ in range(neg_scores.size(0))]).type(FType).to(device)

        scores = torch.cat([pos_scores, neg_scores])
        labels = torch.cat([pos_label, neg_label])
        scores /= self.temp

        loss = -(F.log_softmax(scores, dim=0) * labels).sum() / labels.sum()
        return loss, base_region_emb, neg_region_emb

    def model_train(self, region_id):

        _, flow_matrix = self.ssl_data.get_region(region_id)
        pos_flow_sets = self.positive_sampling(region_id)
        neg_flow_sets = self.negative_sampling(region_id)

        flow_loss, base_region_emb, neg_region_emb = self.forward(flow_matrix, pos_flow_sets, neg_flow_sets)

        return flow_loss, base_region_emb, neg_region_emb

    def get_emb(self):
        output = []
        for region_id in self.ssl_data.sampling_pool:
            _, flow_matrix = self.ssl_data.get_region(region_id)
            region_emb = self.agg_region_emb(flow_matrix)

            output.append(region_emb.detach().cpu().numpy())
        return np.array(output)