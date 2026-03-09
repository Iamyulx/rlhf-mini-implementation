import torch
import torch.nn as nn


class MiniLM(nn.Module):

    def __init__(self, vocab, embed=128):

        super().__init__()

        self.embed = nn.Embedding(vocab, embed)

        self.lstm = nn.LSTM(embed,256,batch_first=True)

        self.fc = nn.Linear(256,vocab)

    def forward(self,x):

        x = self.embed(x)

        out,_ = self.lstm(x)

        return self.fc(out)
