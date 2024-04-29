import torch
import torch.optim as optim
import os
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, SubsetRandomSampler

# 입력 데이터의 형태 2x6x1500
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(2, 32, kernel_size=2, padding=1)  # 패딩 추가
        self.conv2 = nn.Conv2d(32, 64, kernel_size=2, padding=1)  # 패딩 추가
        self.conv3 = nn.Conv2d(64, 128, kernel_size=2, padding=1)  # 패딩 추가
        self.conv4 = nn.Conv2d(128, 256, kernel_size=2, padding=1)
        self.conv5 = nn.Conv2d(256, 512, kernel_size=2, padding=1)

        self.pool = nn.MaxPool2d(2, 2)

        # Global average pooling 레이어 추가
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 4)  # 수정된 부분
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        x = self.pool(F.relu(self.conv5(x)))

        x = self.global_pool(x)  # Global average pooling

        x = x.view(-1, 512)  # Flatten

        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))  # 수정된 부분
        x = self.dropout(x)
        x = self.fc3(x)  # 수정된 부분
        return x


class CustomDataset(Dataset):
    def __init__(self, root_dir, files, labels, transform=None):
        self.root_dir = root_dir
        self.files = files
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file = self.files[idx]
        label = self.labels[idx]
        tensor = torch.load(os.path.join(self.root_dir, file))
        if self.transform:
            tensor = self.transform(tensor)
        # Add Gaussian noise
        # noise = torch.randn_like(tensor) * 0.001  # Adjust std_dev as needed
        # tensor += noise
        return tensor, label


def load_data(prefix_dir, classes, date_dirs):
    real_validation_files = []
    real_validation_labels = []

    for date_dir in date_dirs:
        for i, class_ in enumerate(classes):
            class_dir = os.path.join(f'.\\{prefix_dir}\\{date_dir}\\', '1800', class_)
            if not os.path.exists(class_dir):
                continue
            print(class_dir, '로드 완료. ( 검증셋 )')
            for file in os.listdir(class_dir):
                real_validation_files.append(os.path.join('1800', class_, file))
                real_validation_labels.append(i)

    return real_validation_files, real_validation_labels


def learning_start(set_epoch, set_batch_size, rpm_list):
    # 데이터 로드
    prefix_dir = 'data'
    classes = ['inner', 'normal', 'outer', 'roller']
    set_epoch = set_epoch

    # 학습 데이터 셋 추가하기 =========================================
    data_dir = f'.\\{prefix_dir}\\240201\\'
    dataset_rpm = rpm_list
    files = []
    labels = []

    for rpm_level in dataset_rpm:
        for i, class_ in enumerate(classes):
            class_dir = os.path.join(data_dir, rpm_level, class_)
            if not os.path.exists(class_dir):
                continue
            print(class_dir, '로드 완료. ( 학습 셋 )')
            for file in os.listdir(class_dir):
                files.append(os.path.join(rpm_level, class_, file))
                labels.append(i)


    # 데이터 분리
    train_files, test_files, train_labels, test_labels = train_test_split(files, labels, test_size=0.2, random_state=42)
    train_dataset = CustomDataset(data_dir, train_files, train_labels, transform=None)
    test_dataset = CustomDataset(data_dir, test_files, test_labels, transform=None)


    # 학습 데이터셋과 테스트 데이터셋의 크기
    train_size = len(train_dataset)
    test_size = len(test_dataset)

    # 미니배치 크기 설정
    batch_size = set_batch_size

    # 학습 데이터셋과 테스트 데이터셋의 인덱스를 섞음
    train_indices = np.random.choice(train_size, size=int(train_size * 0.5), replace=False)
    test_indices = np.random.choice(test_size, size=int(test_size * 0.5), replace=False)

    # SubsetRandomSampler를 사용하여 미니배치 학습을 위한 데이터로더 생성
    train_sampler = SubsetRandomSampler(train_indices)
    test_sampler = SubsetRandomSampler(test_indices)

    # 데이터로더 생성
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, sampler=test_sampler)


    print('====================학습 데이터 셋 로드 완료 ==========================')

    # 모델 생성
    model = Net()

    # 손실함수와 옵티마이저 정의
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    # GPU 사용 설정
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)


    for epoch in range(set_epoch):
        print('epoch: #%d' % (epoch + 1))
        running_loss = 0.0

        for i, data in enumerate(train_dataloader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            inputs = inputs.float()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % 200 == 99:
                print('[epoch : %d,  step : %5d  / %5d ] loss: %.3f' % (
                epoch + 1, i + 1, len(train_dataloader), running_loss / (i + 1)))
                running_loss = 0.0

        with torch.no_grad():
            total_correct = 0
            total_samples = 0
            for data in test_dataloader:
                inputs, labels = data[0].to(device), data[1].to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                total_samples += labels.size(0)
                total_correct += (predicted == labels).sum().item()

            accuracy = total_correct / total_samples
            print('[test_dataloader][Epoch %d] Accuracy on validation data: %.2f%%' % (epoch + 1, accuracy * 100))



    # if rpm_list == ['600', '900', '1500']:
    #     base_path = f"./model0409/{prefix_dir}_model_6915_{set_epoch}_{set_batch_size}"
    # elif rpm_list == ['600', '900', '1500', 'increase']:
    #     base_path = f"./model0409/{prefix_dir}_model_6915Increase_{set_epoch}_{set_batch_size}"
    # elif rpm_list == ['600', '900', '1200', '1500', '1800', 'increase']:
    #     base_path = f"./model0409/{prefix_dir}_model_ALL_{set_epoch}_{set_batch_size}"

    base_path =  f"./model0409/{prefix_dir}_model_618_{set_epoch}_{set_batch_size}"
    extension = ".pt"
    index = 1

    # 생성될 파일 경로 결정
    path = f"{base_path}{extension}"

    # 이미 파일이 존재하는 경우 인덱스를 증가시킴
    while os.path.exists(path):
        path = f"{base_path}{index}{extension}"
        index += 1

    # 모델 저장
    torch.save(model.state_dict(), path)

    print(f"모델이 '{path}'에 저장되었습니다.")



def __main__():
    batch_list = [4]
    ep_list = [50, 100, 200, 300]
    rpm_list = ['600', '1800']
    #rpm_list2 = ['600', '900', '1500', '']
    #rpm_list3 = ['600', '900', '1200', '1500', '1800', 'increase']
    #rpm_list3 = ['increase']
    for batch in batch_list:
        for ep in ep_list:
            #learning_start(ep, batch, rpm_list)
            #learning_start(ep, batch, rpm_list2)
            learning_start(ep, batch, rpm_list)


__main__()