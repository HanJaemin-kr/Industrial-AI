import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import os
import torchvision.transforms as transforms
from torch.utils.data import ConcatDataset

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(2, 32, kernel_size=2, padding=1)  # 패딩 추가
        self.conv2 = nn.Conv2d(32, 64, kernel_size=2, padding=1)  # 패딩 추가
        self.conv3 = nn.Conv2d(64, 128, kernel_size=2, padding=1)  # 패딩 추가
        self.conv4 = nn.Conv2d(128, 256, kernel_size=2, padding=1)  # 패딩 추가

        self.pool = nn.MaxPool2d(2, 2)

        # Global average pooling 레이어 추가
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        self.fc1 = nn.Linear(256, 64)
        self.fc2 = nn.Linear(64, 4)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))

        x = self.global_pool(x)  # Global average pooling

        x = x.view(-1, 256)  # Flatten

        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
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
        return tensor, label


def load_data(prefix_dir, classes, data_dirs, model, device, show_detail):

    for data_dir in data_dirs:
        target_dir = os.path.join(prefix_dir, data_dir)
        real_validation_files = []
        real_validation_labels = []

        num_classes = len(classes)  # 클래스의 총 개수를 수정
        class_predicted = {i: [] for i in range(num_classes)}
        total_correct = 0
        total_samples = 0

        for i, class_ in enumerate(classes):
            if(data_dirs == rpm_list):
                setRpm = '.'
            else:
                setRpm = '1800'

            class_dir = os.path.join(prefix_dir, data_dir, setRpm, class_)

            if not os.path.exists(class_dir):
                continue
            if show_detail: (f'{class_dir} 로드 완료.')
            for file in os.listdir(class_dir):
                real_validation_files.append(os.path.join(setRpm, class_, file))
                real_validation_labels.append(i)

        validation_dataset = CustomDataset(target_dir, real_validation_files, real_validation_labels, transform=None)
        validation_dataloader = DataLoader(validation_dataset, batch_size=4, shuffle=False)

        with torch.no_grad():
            for data in validation_dataloader:
                inputs, labels = data[0].to(device), data[1].to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)

                total_samples += labels.size(0)
                total_correct += (predicted == labels).sum().item()

                # 각 클래스에 대한 예측 결과 저장
                for i in range(labels.size(0)):
                    true_label = labels[i].item()
                    predicted_label = predicted[i].item()
                    class_predicted[true_label].append(predicted_label)

        # 정확도 출력
        accuracy = total_correct / total_samples
        print(f'[{class_dir}] - Total samples: {total_samples}, Correct predictions: {total_correct}, Accuracy: {accuracy * 100:.2f}%')

        # 각 클래스별로 값의 등장 횟수 계산
        object_value_counts = {}
        for key, values_list in class_predicted.items():
            value_counts = {}
            for value in values_list:
                if value in value_counts:
                    value_counts[value] += 1
                else:
                    value_counts[value] = 1
            object_value_counts[key] = value_counts

        # 결과 출력
        if(show_detail):
            for key, value_counts in object_value_counts.items():
                print(f'========= 객체 {classes[key]}의 오답 분석: =========')
                for value, count in value_counts.items():
                    if key != value:
                        print(f'  값 {classes[value]}: {count} / {total_samples/4}번')
                print('  ====================================  ')

    return 'success'


def load_model(target_dir, target_name):

    model = Net()
    model.load_state_dict(torch.load( f"{target_dir}/{target_name}"))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    return model, device

rpm_list = ['600', '900', '1200', '1500', '1800']
def __main__():
    classes = ['inner', 'normal', 'outer', 'roller']
    model_dir = './model3'
    pt_files = [file for file in os.listdir(model_dir) if file.endswith('.pt')]

z
    dataset = 'train'
    #dataset = 'test'
    show_detail = False

    print('실행 경로 :  ', os.path.realpath(__file__))
    for model_name in pt_files:
        model, device = load_model(model_dir, model_name)
        print(F"===== ' {model_name} ' 의 결과입니다 =================")

        if(dataset== 'train'):
            prefix_dir = './data/240201'
            load_data(prefix_dir, classes, rpm_list, model, device, show_detail=show_detail)
        else:
            prefix_dir = './valid'
            data_dirs =  ['240310', '240311', '240131','240118', '231025', '231026']
            load_data(prefix_dir, classes, data_dirs, model, device, show_detail=show_detail)
        print('\n')


__main__()
