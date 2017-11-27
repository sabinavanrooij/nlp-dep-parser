import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)

# Initialize the lstm nn, still with random dummy data
lstm = nn.LSTM(150, 150)

# Random dummy inputs
inputs = [autograd.Variable(torch.randn((1, 150)))
          for _ in range(10)]


class LSTMmodel(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, tagset_size):
        super(LSTMmodel, self).__init__()
        self.hidden_dim = hidden_dim

        # LSTM input word/POS embedding, output states
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)

        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        return (autograd.Variable(torch.zeros(1, 1, self.hidden_dim)),
                autograd.Variable(torch.zeros(1, 1, self.hidden_dim)))

    def forward(self, sentence):
        lstm_out, self.hidden = self.lstm(sentence.view(len(sentence), 1, -1), self.hidden)
        tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
        return F.log_softmax(tag_space)


inputs = torch.cat(inputs).view(len(inputs), 1, -1)
model = LSTMmodel(150, 150, 150)
scores = model(inputs)
print(scores)

