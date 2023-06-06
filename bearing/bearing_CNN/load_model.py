from numpy import fromfile
import numpy as np

from os import listdir
from os.path import isfile, join

import pandas as pd
import re

from torchvision.io import read_image
import makedataset
import model
import torch
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader, Dataset, random_split
import shutil

import torch.nn.functional as F

from torchsummary import summary

# 모델 로드


GPU_NUM = 0 # 원하는 GPU 번호 입력
device = torch.device(f'cuda:{GPU_NUM}' if torch.cuda.is_available() else 'cpu')

values = torch.tensor(pd.read_csv("D:\\EC_bearing_images\\testdata\\rollertest.csv", header=None).values,dtype=torch.float)

values = torch.unsqueeze(values, dim=0)

model = model.bearingCNN(values)  # 모델 객체 생성
model.load_state_dict(torch.load("models/model_epoch9.pt"))




# y_ = y_.reshape(batchsize,1,1)

output = model.forward(values)
_, output_index = torch.max(output, 1)




print()