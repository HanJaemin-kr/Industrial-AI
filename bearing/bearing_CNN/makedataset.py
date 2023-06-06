from torch.utils.data import TensorDataset, DataLoader, Dataset, random_split
from os import listdir
import numpy as np
from os.path import isfile, join
import re
from numpy import fromfile
from PIL import Image
import pandas as pd
from matplotlib import pyplot as plt
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'


class getDataset(Dataset):
    # 폴더 리스트, 누출 시간 입력
    def __init__(self, rootFolder):
        super(getDataset, self).__init__()

        # folder : 0.7mm
        self.rootFolder = rootFolder
        self.folders = listdir(rootFolder)

        self.folderNameList = np.array([])
        self.labelList = np.array([])

        # 0.7mm면 0, 1mm면 1, 3mm면 2, good이면 3

        # folder 0.7mm
        for folder in self.folders:
            subfolders = join(rootFolder, folder)
            subfolders = listdir(subfolders)
            subfolders = sorted(subfolders, key=lambda s: int(re.search(r'\d+', s).group()))

            for subfolder in subfolders:
                self.folderNameList = np.append(self.folderNameList, join(rootFolder, folder,subfolder))
                if (folder == "inner"):
                    label = 0

                elif (folder == "outer"):
                    label = 1

                elif (folder == "roller"):
                    label = 2


                self.labelList = np.append(self.labelList, label)
        self.labelList = self.labelList.astype(np.int64)

    def __getitem__(self, index):
        # 정상, 0.7mm, 1mm, 3mm

        signal = pd.read_csv(self.folderNameList[index],header=None).values
        # return 값은 4 x 1000000 배열, labelList[idx]

        return signal, self.labelList[index]

    def __len__(self):
        return len(self.labelList)