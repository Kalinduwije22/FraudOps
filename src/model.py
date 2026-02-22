import torch
import torch.nn as nn

class FraudNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(FraudNet, self).__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.batch_norm = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(0.3)
        self.layer2 = nn.Linear(hidden_dim, int(hidden_dim/2))
        self.output = nn.Linear(int(hidden_dim/2), output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.batch_norm(x)
        x = self.dropout(x)
        x = self.layer2(x)
        x = self.relu(x)
        return self.sigmoid(self.output(x))