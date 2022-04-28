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


class Flatten(torch.nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class POI_SSL(torch.nn.Module):

    def __init__(self, ssl_data, neg_size, emb_size, attention_size, temp, extractor): 
        super(POI_SSL,self).__init__()

        self.ssl_data = ssl_data
        self.init_basic_conf(neg_size, emb_size, attention_size, temp, extractor)
        
    def init_basic_conf(self, neg_size, emb_size, attention_size, temp, extractor):
        self.neg_size = neg_size
        self.emb_size = emb_size
        self.attention_size = attention_size
        self.bin_num = 10

        self.poi_num = self.ssl_data.poi_num
        self.temp = temp

        self.extractor = extractor

        self.W_poi = None

        if self.extractor == "CNN":
            self.poi_net = torch.nn.Sequential(
                nn.Conv1d(in_channels=1, out_channels=4, kernel_size=7),
                Flatten(),
                nn.Linear(964, self.emb_size)
            ).to(device)

        if self.extractor == "MLP":
            self.poi_net = torch.nn.Sequential(
                nn.Linear(247, self.emb_size),
                nn.ReLU(),
            ).to(device)

        if self.extractor == "SA":
            self.poi_net = SAEncoder(d_input=247, d_model=16, n_head=3).to(device)


    def generate_attention(self, W_parents, W_children, mask):
        _input = torch.cat([W_parents, W_children], dim=2)
        output = self.tree_att_net(_input)
        pre_attention = torch.matmul(output, self.v_attention)

        pre_attention = pre_attention + mask
        attention = torch.softmax(pre_attention, dim=1)
        
        return attention

    def tree_gcn(self):
        parentsList, childrenList, maskList = [], [], []

        for i, p2c_i in enumerate(self.ssl_data.level_p2c[::-1]):
            children = list(p2c_i.values())
            max_n_children = max(len(x) for x in children)
            mask = []

            for k in children:
                cur_mask = [0.0] * len(k)
                if len(k) < max_n_children:
                    cur_mask += [-10**13] * (max_n_children - len(k))
                    k += [0] * (max_n_children - len(k))
                mask.append(cur_mask)

            parents = []
            for p in p2c_i.keys():
                parents.append([p] * max_n_children)

            children = torch.Tensor(children).type(LType)
            parents = torch.Tensor(parents).type(LType)
            mask = torch.Tensor(mask).type(FType)
            
            parentsList.append(parents)
            childrenList.append(children)
            maskList.append(mask)

        W_emb_temp = self.W_poi.clone() + 0.
        for i, (parents, children, mask) in enumerate(zip(parentsList, childrenList, maskList)):
            W_parents = self.W_poi[parents]
            if i == 0:
                W_children = self.W_poi[children]
            else:
                W_children = W_emb_temp[children]

            tempAttention = self.generate_attention(W_parents, W_children, mask)
            tempEmb = (W_children * tempAttention[:,:,None]).sum(dim=1)

            W_emb_temp = torch.index_copy(W_emb_temp, 0, parents[:, 0], tempEmb)

        parentsList, childrenList, maskList = [], [], []
        for i, c2p_i in enumerate(self.ssl_data.level_c2p[::-1]):
            parents = list(c2p_i.values())
            max_n_parents = max(len(x) for x in parents)
            mask = []

            for k in parents:
                cur_mask = [0.0] * len(k)
                if len(k) < max_n_parents:
                    cur_mask += [-10**13] * (max_n_parents - len(k))
                    k += [0] * (max_n_parents - len(k))
                mask.append(cur_mask)

            children = []
            for c in c2p_i.keys():
                children.append([c] * max_n_parents)

            children = torch.Tensor(children).type(LType)
            parents = torch.Tensor(parents).type(LType)
            mask = torch.Tensor(mask).type(FType)
            
            parentsList.append(parents)
            childrenList.append(children)
            maskList.append(mask)

        for i, (parents, children, mask) in enumerate(zip(parentsList, childrenList, maskList)):
            W_children, W_parents = W_emb_temp[children], W_emb_temp[parents]

            tempAttention = self.generate_attention(W_children, W_parents, mask)
            # tempEmb = (W_parents * tempAttention[:,:,None]).sum(axis=1)
            tempEmb = (W_parents * tempAttention[:,:,None]).sum(dim=1)

            # W_emb_temp[children[:, 0]] = tempEmb
            W_emb_temp = torch.index_copy(W_emb_temp, 0, children[:, 0], tempEmb)

        self.W_emb_temp = W_emb_temp

        return W_emb_temp

    def location_attention(self, loc_emb_one, loc_emb_two):
        _input = torch.cat([loc_emb_one, loc_emb_two], axis=1)

        output = self.location_att_net(_input)
        pre_attention = torch.matmul(output, self.l_attention)

        attention = torch.softmax(pre_attention, dim=0)
        return attention

    def agg_region_emb(self, poi_set, W_emb_temp):
        p_node_dict = self.ssl_data.region_dict[0]["node_dict"] 
        poi_f = np.zeros(len(p_node_dict))
        for poi in poi_set:
            poi_id = poi[0]
            poi_f[poi_id] += 1

        if np.sum(poi_f) != 0:
            poi_f = poi_f / np.sum(poi_f)
        poi_f = torch.Tensor(poi_f).type(FType).to(device)

        if self.extractor == "CNN":
            poi_f = poi_f.unsqueeze(0)
            poi_f = poi_f.unsqueeze(0)
            temp_emb = self.poi_net(poi_f)

        if self.extractor == "MLP":
            temp_emb = self.poi_net(poi_f)

        if self.extractor == "SA":
            temp_emb = self.poi_net(poi_f)

        region_emb = temp_emb
        region_emb = region_emb.squeeze()

        return region_emb

    def add_aug(self, poi_set, _ratio):
        add_poi_set = []
        for poi in poi_set:
            add_poi_set.append(poi)
            ratio = random.random()
            if ratio < _ratio:
                add_poi_set.append(poi)
        return add_poi_set

    def delete_aug(self, poi_set, _ratio):
        de_poi_set = []
        for poi in poi_set:
            ratio = random.random()
            if ratio > _ratio:
                de_poi_set.append(poi)
        if not de_poi_set:
            de_poi_set = [poi_set[0]]
        return de_poi_set

    def replace_aug(self, poi_set, _ratio):
        replace_poi_set = []
        for poi in poi_set:
            new_poi = poi
            ratio = random.random()
            if ratio < _ratio:
                new_poi[0] = random.randint(0, self.ssl_data.poi_num-1)
            replace_poi_set.append(new_poi)
        return replace_poi_set

    def positive_sampling(self, region_id):
        pos_poi_sets = []
        poi_set, _ = self.ssl_data.get_region(region_id)

        de_poi_set = []
        for ratio in [0.1]:
            de_poi_set.append(self.delete_aug(poi_set, ratio))

        add_poi_set = []
        for ratio in [0.1]:
            add_poi_set.append(self.add_aug(poi_set, ratio))

        re_poi_set = []
        for ratio in [0.1]:
            re_poi_set.append(self.replace_aug(poi_set, ratio))
        
        pos_poi_sets = de_poi_set + add_poi_set + re_poi_set

        return pos_poi_sets

    def negative_sampling(self, region_id):
        sampling_pool = []
        for _id in self.ssl_data.sampling_pool:
            if _id == region_id:
                continue
            sampling_pool.append(_id)

        p = self.ssl_data.rs_ratio["model_poi"][region_id]
        neg_region_ids = np.random.choice(sampling_pool, self.neg_size, replace=False, p=p)

        neg_poi_sets = []
        for neg_region_id in neg_region_ids:
            poi_set, _ = self.ssl_data.get_region(neg_region_id)
            neg_poi_sets.append(poi_set)

        return neg_poi_sets

    def forward(self, poi_set, pos_poi_sets, neg_poi_sets):
        
        # W_emb_temp = self.tree_gcn()
        W_emb_temp = self.W_poi

        base_region_emb = self.agg_region_emb(poi_set, W_emb_temp)

        pos_region_emb_list = []
        for pos_poi_set in pos_poi_sets:
            pos_region_emb = self.agg_region_emb(pos_poi_set, W_emb_temp)
            pos_region_emb_list.append(pos_region_emb.unsqueeze(0))
        pos_region_emb = torch.cat(pos_region_emb_list, dim=0)

        neg_region_emb_list = []
        for neg_poi_set in neg_poi_sets:
            neg_region_emb = self.agg_region_emb(neg_poi_set, W_emb_temp)
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
        poi_set, _ = self.ssl_data.get_region(region_id)
        pos_poi_sets = self.positive_sampling(region_id)
        neg_poi_sets = self.negative_sampling(region_id)

        poi_loss, base_region_emb, neg_region_emb = self.forward(poi_set, pos_poi_sets, neg_poi_sets)

        return poi_loss, base_region_emb, neg_region_emb 

    def get_emb(self):
        output = []
        for region_id in self.ssl_data.sampling_pool:
            poi_set, _ = self.ssl_data.get_region(region_id)
            region_emb = self.agg_region_emb(poi_set, self.W_poi)

            output.append(region_emb.detach().cpu().numpy())
        return np.array(output)
