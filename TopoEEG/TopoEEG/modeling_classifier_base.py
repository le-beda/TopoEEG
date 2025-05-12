import torch.nn as nn


class Classifier(nn.Module):
    def __init__(self, 
                 in_features,
                 out_features
                 ):

        super(Classifier, self).__init__()

        self.out_features=out_features

        self.fc1 = nn.Linear(in_features=in_features, out_features=32)
        self.bn1 = nn.BatchNorm1d(num_features=32)
        self.relu = nn.ReLU()
        self.dropout1 = nn.Dropout(p=0.5)

        self.fc2 = nn.Linear(in_features=32, out_features=out_features)
        self.bn2 = nn.BatchNorm1d(num_features=out_features)
        self.dropout2 = nn.Dropout(p=0.5)

        # self.fc3 = nn.Linear(in_features=8, out_features=1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout1(x)

        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout2(x)

        # x = self.fc3(x)

        return x
