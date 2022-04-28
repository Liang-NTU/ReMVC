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

FType = torch.FloatTensor
LType = torch.LongTensor


region_dict_path = "./clear_data/training_dict.pickle"
region2poi_path = "./clear_data/region2poi.pickle"
rs_ratio_path = "./clear_data/rs_ratio.pickle"

from configparser import ConfigParser
config = ConfigParser()
config.read('conf', encoding='UTF-8')
GPU_DEVICE = config['DEFAULT'].get("GPU_DEVICE")
device = torch.device("cuda:"+GPU_DEVICE if torch.cuda.is_available() else "cpu")

class SSLData:

    def __init__(self):

        self.region_dict = pickle.load(open(region_dict_path, "rb"))
        self.sampling_pool = [_i for _i in range(len(self.region_dict))]

        region2poi = pickle.load(open(region2poi_path, "rb"))
        self.level_c2p = region2poi["level_c2p"]
        self.level_p2c = region2poi["level_p2c"]
        self.poi_num = len(region2poi["node_dict"])

        self.rs_ratio = pickle.load(open(rs_ratio_path, "rb"))

    def get_region(self, idx):
        pois = self.region_dict[idx]["poi"]

        poi_set = []
        for poi in pois:
            _id = poi[1]
            l_vector = [poi[2].index(1), poi[3].index(1)]
            poi_set.append([_id, l_vector])

        pickup_matrix = self.region_dict[idx]["pickup_matrix"]
        dropoff_matrix = self.region_dict[idx]["dropoff_matrix"]

        pickup_matrix = pickup_matrix / pickup_matrix.sum()
        where_are_NaNs = np.isnan(pickup_matrix)
        pickup_matrix[where_are_NaNs] = 0

        dropoff_matrix = dropoff_matrix / dropoff_matrix.sum()
        where_are_NaNs = np.isnan(dropoff_matrix)
        dropoff_matrix[where_are_NaNs] = 0

        flow_matrix = [pickup_matrix, dropoff_matrix]

        return poi_set, flow_matrix