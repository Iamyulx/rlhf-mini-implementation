import torch
import torch.nn as nn


class RewardModel(nn.Module):

    def __init__(self, vocab):

        super().__init__()

        self.embed = nn.Embedding(vocab,128)

        self.lstm = nn.LSTM(128,256,batch_first=True)

        self.score = nn.Linear(256,1)

    def forward(self,x):

        x = self.embed(x)

        out,_ = self.lstm(x)

        last = out[:,-1,:]

        return self.score(last)
