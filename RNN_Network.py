import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math


class SelfAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(SelfAttention, self).__init__()
        self.Wk = nn.Linear(d_model, d_model)
        self.Wq = nn.Linear(d_model, d_model)
        self.V = nn.Linear(d_model, d_model)
        self.Wh = nn.Linear(d_model, d_model)
        self.head = num_heads
        self.d_k = d_model // self.head

    def forward(self, x):
        batch_size = x.shape[0]
        query = self.Wq(x)
        key = self.Wk(x)
        value = self.V(x)

        Q = query.view(batch_size, -1, self.head, self.d_k).transpose(1, 2)
        K = key.view(batch_size, -1, self.head, self.d_k).transpose(1, 2)
        V = value.view(batch_size, -1, self.head, self.d_k).transpose(1, 2)

        attention_logits = torch.matmul(Q, K.transpose(-2, -1))
        attention_weights = F.softmax(attention_logits / math.sqrt(self.d_k), dim=-1)
        attention_output = torch.matmul(attention_weights, V)

        attention_output = attention_output.transpose(1, 2).contiguous().view(batch_size, -1, self.head * self.d_k)
        attention_output = self.Wh(attention_output)

        return attention_output, attention_weights


class RNNAttention(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, num_heads, n_layers=1,
                 bidirectional=True, dropout=0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, num_layers=n_layers, bidirectional=bidirectional, dropout=dropout,
                           batch_first=True)
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.attention = SelfAttention(embedding_dim, num_heads)

    def forward(self, text):
        embedded = self.embedding(text)
        attn_output, attn_weight = self.attention(embedded)
        lstm_output, (hidden, cell) = self.rnn(attn_output)
        hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        output = torch.relu(self.fc1(hidden))
        output = self.fc2(output)

        return output
