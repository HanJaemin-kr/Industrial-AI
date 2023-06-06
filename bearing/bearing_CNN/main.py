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


bearing = "D:/EC_bearing_images"


# 데이터셋 구성**************************************************************



bearingDataSet = makedataset.getDataset(bearing)

dataset_size = len(bearingDataSet)
train_size = int(dataset_size * 0.5)
test_size = dataset_size - train_size
#validation_size = dataset_size - train_size - test_size

train_dataset, test_dataset = random_split(bearingDataSet, [train_size, test_size])
# GPU 설정

GPU_NUM = 0 # 원하는 GPU 번호 입력
device = torch.device(f'cuda:{GPU_NUM}' if torch.cuda.is_available() else 'cpu')
# torch.cuda.set_device(device)

# if torch.cuda.is_available():
#     torch.set_default_tensor_type(torch.cuda.FloatTensor)
#     print(f"using cuda: {GPU_NUM}, {torch.cuda.get_device_name(GPU_NUM)}")

batchsize = 2
learning_rate = 0.0002
num_epoch = 5

train_dataloader = DataLoader(train_dataset, batch_size=batchsize, shuffle=True,drop_last=True)
test_dataloader = DataLoader(test_dataset, batch_size=batchsize, shuffle=True,drop_last=True)



for [data, label] in train_dataloader:
    model = model.bearingCNN(data[1,:,:].unsqueeze(0).float())
    break

loss_func = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
# 학습
for i in range(10):
    loss_avg=0
    cnt = 0
    for seq, [values, labels] in enumerate(train_dataloader):
        x = values.float().to(device)
        y_ = labels.long().to(device)

        # x = x.reshape(32,1,4,1000000)
        # print(x.shape)
        # print(y_.shape)

        optimizer.zero_grad()
        output = model.forward(x)

        # break
        # print("for문 output 크기",output.shape)
        # print()

        loss = loss_func(output,y_)
        loss.backward()
        loss_avg += loss.detach().cpu().numpy()
        cnt += 1
        optimizer.step()

        # break
    loss_avg /= cnt
    print("{} epochs, loss : {}".format(i, str(loss_avg)))

    torch.save(model.state_dict(), "models/model_epoch{}.pt".format(i))

del train_dataloader # 학습 데이터 삭제
torch.cuda.empty_cache() # GPU 캐시 데이터 삭제


# 테스트
correct = 0
total = 0

# 배치정규화나 드롭아웃은 학습할때와 테스트 할때 다르게 동작하기 때문에 모델을 evaluation 모드로 바꿔서 테스트해야합니다.
model.eval()
with torch.no_grad():
    for values,labels in test_dataloader:

        x = values.float().to(device)
        y_= labels.long().to(device)

        #y_ = y_.reshape(batchsize,1,1)


        output = model.forward(x)
        _,output_index = torch.max(output,1)

        total += labels.size(0)
        correct += (output_index == y_).sum().float()

print("Accuracy of Test Data: {}".format(100*correct/total))