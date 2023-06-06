from torch import nn

class bearingCNN(nn.Module):
    def __init__(self,dummy):
        # super함수는 CNN class의 부모 class인 nn.Module을 초기화
        super(bearingCNN, self).__init__()

        self.conv_layer1 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=4, kernel_size=4, stride=1),
            nn.LeakyReLU(),
            nn.BatchNorm1d(4),
            # nn.Dropout(0.25),
            nn.MaxPool1d(2),

            nn.Conv1d(in_channels=4, out_channels=8, kernel_size=4, stride=1),
            nn.LeakyReLU(),
            nn.BatchNorm1d(8),
            # nn.Dropout(0.25),
            nn.MaxPool1d(2),

            nn.Conv1d(in_channels=8, out_channels=4, kernel_size=4, stride=1),
            nn.LeakyReLU(),
            nn.BatchNorm1d(4),
            # nn.Dropout(0.25),
            nn.MaxPool1d(2),


        )

        self.conv_layer2 = nn.Sequential(
            nn.Conv1d(in_channels=4, out_channels=2, kernel_size=4, stride=1),
            nn.LeakyReLU(),
            nn.BatchNorm1d(2),
            # nn.Dropout(0.25),
            nn.MaxPool1d(2)

        )
        dummyout = self.conv_layer1(dummy)
        dummyout = self.conv_layer2(dummyout)
        outputsize = dummyout.view(-1).shape[0]

        self.fc_layers = nn.Sequential(
            # 여기 들어갈 입력 갯수는 미리 정해ㅝ야함
            nn.Linear(outputsize, 10),
            nn.Tanh(),
            nn.Linear(10, 4),
            nn.Softmax(dim=1),
        )

    def forward(self, x):
        # self.layer에 정의한 연산 수행

        out = self.conv_layer1(x)
        # print("layer1 result :         ", out.shape)

        out = self.conv_layer2(out)
        # print("layer2 result :         ", out.shape)

        out = out.view(x.shape[0],-1)
        # print("lyaer2.view result :    ", out.shape)

        out = self.fc_layers(out)

        # print("fc result :             ", out.shape)

        return out
